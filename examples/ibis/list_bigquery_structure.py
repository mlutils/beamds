#!/usr/bin/env python3
"""
List all datasets and tables in the algo-agents-ai21 BigQuery project
"""

from beam import resource
from google.cloud import bigquery


def list_project_structure():
    """List all datasets and tables in the BigQuery project."""
    
    print("🏢 BigQuery Project: algo-agents-ai21")
    print("=" * 50)
    
    try:
        # Connect using beam resource system
        bi = resource('ibis-bigquery:///algo-agents-ai21')
        print(f"✅ Connected successfully!")
        print(f"   Backend: {bi.backend}")
        print(f"   Level: {bi.level}")
        
        # Get BigQuery client for project exploration
        client = bigquery.Client(project='algo-agents-ai21')
        
        # List all datasets
        datasets = list(client.list_datasets())
        print(f"\n📊 Found {len(datasets)} datasets:")
        
        total_tables = 0
        
        for i, dataset in enumerate(datasets, 1):
            dataset_id = dataset.dataset_id
            dataset_ref = client.dataset(dataset_id)
            
            # List tables in this dataset
            tables = list(client.list_tables(dataset_ref))
            table_count = len(tables)
            total_tables += table_count
            
            print(f"\n{i}. 📂 {dataset_id}")
            print(f"   Tables: {table_count}")
            
            if tables:
                for j, table in enumerate(tables, 1):
                    table_type = "📊" if table.table_type == "TABLE" else "👁️"
                    print(f"   {j:2d}. {table_type} {table.table_id}")
                    
                    # Get basic table info
                    try:
                        table_ref = dataset_ref.table(table.table_id)
                        table_obj = client.get_table(table_ref)
                        
                        # Show row count and schema info
                        row_count = table_obj.num_rows
                        col_count = len(table_obj.schema)
                        
                        print(f"       └── {row_count:,} rows, {col_count} columns")
                        
                        # Show first few column names
                        if col_count > 0:
                            col_names = [field.name for field in table_obj.schema[:3]]
                            col_preview = ", ".join(col_names)
                            if col_count > 3:
                                col_preview += f", ... (+{col_count-3} more)"
                            print(f"       └── Columns: {col_preview}")
                            
                    except Exception as e:
                        print(f"       └── Could not get table details: {e}")
            else:
                print("   (No tables)")
        
        print(f"\n" + "=" * 50)
        print(f"📈 Summary:")
        print(f"   • Total datasets: {len(datasets)}")
        print(f"   • Total tables: {total_tables}")
        
        return datasets, total_tables
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, 0


def main():
    """Main function."""
    datasets, total_tables = list_project_structure()
    
    if datasets:
        print(f"\n✅ Successfully listed {len(datasets)} datasets and {total_tables} tables!")
    else:
        print(f"\n❌ Failed to list project structure")


if __name__ == "__main__":
    main() 