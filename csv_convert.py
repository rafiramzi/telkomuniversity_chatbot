import csv
import argparse
from pathlib import Path


def escape_sql_string(value: str) -> str:
    """Escape single quotes for SQL by doubling them."""
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def csv_to_sql(csv_path: str, sql_path: str, batch_size: int = 100) -> None:
    """
    Convert CSV file to SQL file.

    Args:
        csv_path: Path to input CSV file
        sql_path: Path to output SQL file
        batch_size: Number of rows per INSERT statement (for performance)
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as cf, open(sql_path, "w", encoding="utf-8") as sf:
        # 1) Write CREATE TABLE statement
        sf.write("-- Auto-generated SQL from CSV\n")
        sf.write("-- Table: datasets\n\n")
        sf.write("CREATE TABLE IF NOT EXISTS datasets (\n")
        sf.write("    id SERIAL PRIMARY KEY,\n")
        sf.write("    file VARCHAR(255) NOT NULL,\n")
        sf.write("    category VARCHAR(100),\n")
        sf.write("    text TEXT,\n")
        sf.write("    created_at TIMESTAMP NOT NULL DEFAULT NOW()\n")
        sf.write(");\n\n")

        # 2) Read CSV and write INSERT statements
        reader = csv.DictReader(cf)

        required_cols = {"file", "category", "text"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            missing = required_cols - set(reader.fieldnames or [])
            raise ValueError(f"CSV is missing required columns: {missing}")

        batch = []
        total = 0

        def flush(batch_rows):
            if not batch_rows:
                return
            sf.write("INSERT INTO datasets (file, category, text, created_at) VALUES\n")
            values_lines = []
            for r in batch_rows:
                values_lines.append(
                    "    ({fn}, {cat}, {txt}, NOW())".format(
                        fn=escape_sql_string(r["file"]),
                        cat=escape_sql_string(r["category"]),
                        txt=escape_sql_string(r["text"]),
                    )
                )
            sf.write(",\n".join(values_lines))
            sf.write(";\n\n")

        for row in reader:
            batch.append(row)
            total += 1
            if len(batch) >= batch_size:
                flush(batch)
                batch = []

        # Flush remaining rows
        flush(batch)

        print(f"✓ Converted {total} rows from {csv_path} to {sql_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert CSV to SQL INSERT statements")
    parser.add_argument("csv_file", help="Path to the input CSV file")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path to output SQL file (default: <csv_name>.sql)"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=100,
        help="Rows per INSERT statement (default: 100)"
    )
    args = parser.parse_args()

    output_path = args.output or str(Path(args.csv_file).with_suffix(".sql"))
    csv_to_sql(args.csv_file, output_path, args.batch_size)


if __name__ == "__main__":
    main()