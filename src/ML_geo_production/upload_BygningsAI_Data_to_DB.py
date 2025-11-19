import os
import sys
import tempfile
import geopandas as gpd
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
import argparse
import configparser



def polygon_to_multipolygon(geom):
    if geom is None:
        return None
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    return geom

def process_shapefiles(shapefile_dir):
    if not os.path.exists(shapefile_dir):
        raise ValueError(f"Directory not found: {shapefile_dir}")

    data_frames = []
    skipped_files = []
    required_columns = ["value", "geometry"]

    for root, dirs, files in os.walk(shapefile_dir):
        for file in files:
            if file.endswith(".shp"):
                shapefile_path = os.path.join(root, file)
                print(f"Processing {shapefile_path}...")

                try:
                    gdf = gpd.read_file(shapefile_path)
                    missing_required = [col for col in required_columns if col not in gdf.columns]
                    if missing_required:
                        print(f"Warning: Skipping {file} - Missing required columns: {', '.join(missing_required)}")
                        skipped_files.append((file, f"Missing required columns: {', '.join(missing_required)}"))
                        continue

                    df = pd.DataFrame()
                    df["value"] = gdf["value"]
                    df["geom"] = gdf["geometry"]
                    df["status"] = gdf["STATUS"] if "STATUS" in gdf.columns else None
                    df["geometrist"] = gdf["GEOMETRIST"] if "GEOMETRIST" in gdf.columns else None
                    df["id_lokalid"] = gdf["ID_LOKALID"] if "ID_LOKALID" in gdf.columns else None
                    df["registreri"] = gdf["REGISTRERI"] if "REGISTRERI" in gdf.columns else None
                    df["label_desc"] = gdf["label_desc"] if "label_desc" in gdf.columns else None
                    df["model_desc"] = gdf["model_desc"] if "model_desc" in gdf.columns else None
                    df["areal"] = gdf["Areal"] if "Areal" in gdf.columns else None

                    df["geom"] = df["geom"].apply(polygon_to_multipolygon)
                    df["geom"] = df["geom"].apply(lambda x: x.wkt if x else None)
                    df["value"] = df["value"].astype(int)

                    if "areal" in df.columns and df["areal"].notna().any():
                        df["areal"] = df["areal"].astype(float).round(2)

                    for col in ["status", "geometrist", "id_lokalid", "registreri", "label_desc", "model_desc"]:
                        if df[col].notna().any():
                            df[col] = df[col].astype(str).apply(lambda x: x[:80])

                    data_frames.append(df)

                except Exception as e:
                    print(f"Warning: Error processing {file}: {str(e)}")
                    skipped_files.append((file, str(e)))
                    continue

    if not data_frames:
        raise ValueError(f"No valid shapefiles found in {shapefile_dir}. All files were skipped.")

    if skipped_files:
        print("\nSkipped files summary:")
        for file, reason in skipped_files:
            print(f"- {file}: {reason}")
        print()

    final_df = pd.concat(data_frames, ignore_index=True)
    final_df["value"] = final_df["value"].astype(int)

    if "areal" in final_df.columns and final_df["areal"].notna().any():
        final_df["areal"] = final_df["areal"].astype(float).round(2)

    for col in ["status", "geometrist", "id_lokalid", "registreri", "label_desc", "model_desc"]:
        if col in final_df.columns and final_df[col].notna().any():
            final_df[col] = final_df[col].astype(str).apply(lambda x: x[:80])

    return final_df

def upload_to_postgres(csv_path,DMZ_user,DMZ_host,DMZ_port,DMZ_schema,DMZ_table,DMZ_name,DMZ_pass):
    db_connection_str = f'postgresql://{DMZ_user}:{DMZ_pass}@{DMZ_host}:{DMZ_port}/{DMZ_name}'
    try:
        engine = create_engine(db_connection_str)
        df = pd.read_csv(csv_path)
        df["value"] = df["value"].astype(int)

        for col in ["status", "geometrist", "id_lokalid", "registreri", "label_desc", "model_desc"]:
            if col in df.columns and df[col].notna().any():
                df[col] = df[col].astype(str).apply(lambda x: x[:80])
            elif col not in df.columns:
                df[col] = None

        if "areal" in df.columns and df["areal"].notna().any():
            df["areal"] = df["areal"].astype(float).round(2)
        elif "areal" not in df.columns:
            df["areal"] = None

        df['geom'] = df['geom'].apply(lambda x: wkt.loads(str(x)) if pd.notnull(x) else None)
        df['geom'] = df['geom'].apply(polygon_to_multipolygon)
        gdf = gpd.GeoDataFrame(df, geometry='geom')

        conn = psycopg2.connect(
            dbname=DMZ_name,
            user=DMZ_user,
            password=DMZ_pass,
            host=DMZ_host,
            port=DMZ_port
        )

        cur = conn.cursor()
        cur.execute(f"SELECT COALESCE(MAX(fid), 0) FROM {DMZ_schema}.{DMZ_table}")
        next_fid = cur.fetchone()[0] + 1
        cur.close()

        inserted_count = 0
        skipped_count = 0
        error_count = 0
        total_records = len(gdf)

        print(f"\nUploading {total_records} records to database...")
        batch_size = 100
        for batch_start in range(0, total_records, batch_size):
            batch_end = min(batch_start + batch_size, total_records)
            cur = conn.cursor()
            try:
                for idx in range(batch_start, batch_end):
                    row = gdf.iloc[idx]
                    if row['geom'] is None:
                        print(f"Skipping record {idx}: NULL geometry")
                        error_count += 1
                        continue

                    check_query = f"""
                        SELECT COUNT(*) FROM {DMZ_schema}.{DMZ_table}
                        WHERE ST_Equals(geom, ST_GeomFromText(%s, 25832))
                    """
                    cur.execute(check_query, (row['geom'].wkt,))
                    exists = cur.fetchone()[0] > 0

                    if not exists:
                        insert_query = f"""
                            INSERT INTO {DMZ_schema}.{DMZ_table} (value, fid, status, geometrist, id_lokalid, registreri, label_desc, model_desc, areal, geom)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, ST_GeomFromText(%s, 25832))
                        """
                        cur.execute(insert_query, (
                            int(row['value']),
                            next_fid + idx,
                            row['status'],
                            row['geometrist'],
                            row['id_lokalid'],
                            row['registreri'],
                            row['label_desc'],
                            row['model_desc'],
                            float(row['areal']) if pd.notnull(row['areal']) else None,
                            row['geom'].wkt
                        ))
                        inserted_count += 1
                    else:
                        skipped_count += 1

                conn.commit()
                print(f"Progress: {batch_end}/{total_records} records processed...")
            except Exception as e:
                print(f"Error processing batch {batch_start}-{batch_end}: {str(e)}")
                conn.rollback()
                error_count += (batch_end - batch_start)
            finally:
                cur.close()

        print(f"\nUpload complete!")
        print(f"- Inserted: {inserted_count} new records")
        print(f"- Skipped: {skipped_count} duplicates")
        print(f"- Errors: {error_count} records")
        print(f"- Total processed: {total_records} records")

    except Exception as e:
        print(f"Error during upload: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals():
            conn.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('shapefile_dir', help='Directory containing shapefiles to process')
    parser.add_argument('--skip-upload', action='store_true', help='Skip uploading to database, only create CSV')
    parser.add_argument('--output-csv', help='Path to save or read the CSV file')
    parser.add_argument('--skip-reading-shp-files', action='store_true', help='Skip reading shapefiles and use CSV directly')
    parser.add_argument('--config', help='Path to custom config.ini file', default = 'config.ini')
    args = parser.parse_args()

    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        # Load database credentials from config.ini
        config = configparser.ConfigParser()
        config.read(args.config)

        DMZ_name = config['DATABASE']['name']
        DMZ_host = config['DATABASE']['host']
        DMZ_port = config['DATABASE']['port']
        DMZ_user = config['DATABASE']['user']
        DMZ_pass = config['DATABASE']['password']
        DMZ_schema = config['DATABASE']['schema']
        DMZ_table = config['DATABASE']['table']


    try:
        if args.skip_reading_shp_files:
            if not args.output_csv or not os.path.exists(args.output_csv):
                print("Error: --skip-reading-shp-files requires --output-csv with an existing CSV file.")
                sys.exit(1)
            csv_path = args.output_csv
        else:
            print(f"Processing shapefiles from: {args.shapefile_dir}")
            df = process_shapefiles(args.shapefile_dir)
            csv_path = args.output_csv if args.output_csv else tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
            df.to_csv(csv_path, index=False)
            print(f"CSV saved to: {csv_path}")

        if not args.skip_upload:
            upload_to_postgres(csv_path,DMZ_user,DMZ_host,DMZ_port,DMZ_schema,DMZ_table,DMZ_name,DMZ_pass)
        else:
            print("Upload skipped.")

        if not args.output_csv and not args.skip_reading_shp_files:
            os.remove(csv_path)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
