#!/usr/bin/env python3
"""
BigQuery Exploration with BeamIbis

This script demonstrates how to connect to BigQuery using the beam resource system
and explore datasets, tables, and data in the algo-agents-ai21 project.
"""

import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import time

# Import the beam resource system
from beam import resource
from google.cloud import bigquery


def create_sample_data():
    """Create sample data for demonstration, similar to beam_ibis_example.py."""
    
    data = []
    base_date = datetime(2024, 1, 1)
    
    for i in range(1000):
        data.append({
            'id': i,
            'timestamp': base_date + timedelta(hours=i),
            'user_id': f"user_{i % 100}",
            'product': f"product_{i % 20}",
            'category': ['electronics', 'books', 'clothing', 'food'][i % 4],
            'price': round(10 + (i % 100) * 0.5, 2),
            'quantity': (i % 10) + 1,
            'revenue': round((10 + (i % 100) * 0.5) * ((i % 10) + 1), 2),
            'country': ['US', 'UK', 'DE', 'FR', 'JP'][i % 5],
            'city': ['New York', 'London', 'Berlin', 'Paris', 'Tokyo'][i % 5],
            'status': ['completed', 'pending', 'cancelled'][i % 3]
        })
    
    return pd.DataFrame(data)


def create_test_dataset_and_tables():
    """Create test_planning dataset and populate it with test tables."""
    
    print("\n" + "="*60)
    print("CREATING TEST DATASET AND TABLES")
    print("="*60)
    
    try:
        # Initialize BigQuery client
        client = bigquery.Client(project='algo-agents-ai21')
        
        # Create dataset
        dataset_id = 'test_planning'
        dataset_ref = client.dataset(dataset_id)
        
        print(f"🏗️  Creating dataset: {dataset_id}")
        
        try:
            # Try to get the dataset first
            dataset = client.get_dataset(dataset_ref)
            print(f"✅ Dataset {dataset_id} already exists")
        except Exception:
            # Dataset doesn't exist, create it
            dataset = bigquery.Dataset(dataset_ref)
            dataset.description = "Test dataset for BeamIbis functionality testing"
            dataset.location = "US"  # or your preferred location
            
            dataset = client.create_dataset(dataset, exists_ok=True)
            print(f"✅ Created dataset: {dataset_id}")
        
        # Create sample data
        print(f"📊 Creating sample data...")
        df = create_sample_data()
        print(f"   Sample data shape: {df.shape}")
        
        # Create tables
        tables_to_create = [
            'sales',
            'transactions',  # Duplicate for testing
            'user_activity'
        ]
        
        for table_name in tables_to_create:
            print(f"\n🏗️  Creating table: {table_name}")
            
            # Create table reference
            table_ref = dataset_ref.table(table_name)
            
            # Define schema based on our sample data
            schema = [
                bigquery.SchemaField("id", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"), 
                bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("product", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("category", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("price", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("quantity", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("revenue", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("country", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("city", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("status", "STRING", mode="NULLABLE"),
            ]
            
            # Create table
            table = bigquery.Table(table_ref, schema=schema)
            table.description = f"Test table for BeamIbis: {table_name}"
            
            try:
                table = client.create_table(table, exists_ok=True)
                print(f"   ✅ Created table: {table_name}")
                
                # Insert sample data
                print(f"   📥 Inserting sample data...")
                
                # Convert timestamp to proper format for BigQuery
                df_copy = df.copy()
                df_copy['timestamp'] = df_copy['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Insert data using pandas-gbq for simplicity
                try:
                    # Alternative: use client.load_table_from_dataframe
                    job_config = bigquery.LoadJobConfig()
                    job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
                    
                    job = client.load_table_from_dataframe(
                        df_copy, table_ref, job_config=job_config
                    )
                    job.result()  # Wait for the job to complete
                    
                    print(f"   ✅ Inserted {len(df_copy)} rows into {table_name}")
                    
                except Exception as e:
                    print(f"   ⚠️  Failed to insert data into {table_name}: {e}")
                    
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"   ✅ Table {table_name} already exists")
                else:
                    print(f"   ❌ Failed to create table {table_name}: {e}")
        
        print(f"\n✅ Test dataset setup complete!")
        return dataset_id
        
    except Exception as e:
        print(f"❌ Failed to create test dataset: {e}")
        return None


def test_beamibis_functionality(dataset_id='test_planning'):
    """Test BeamIbis functionality using the test dataset."""
    
    print("\n" + "="*60)
    print("TESTING BEAMIBIS FUNCTIONALITY")
    print("="*60)
    
    try:
        # Test basic connection
        print(f"🔗 Connecting to test dataset: {dataset_id}")
        dataset_bi = resource(f'ibis-bigquery:///algo-agents-ai21/{dataset_id}')
        print(f"✅ Connected to dataset level")
        print(f"   Level: {dataset_bi.level}")
        
        # Test table connection
        table_name = 'sales'
        print(f"\n📊 Connecting to table: {table_name}")
        table_bi = resource(f'ibis-bigquery:///algo-agents-ai21/{dataset_id}/{table_name}')
        print(f"✅ Connected to table level")
        print(f"   Level: {table_bi.level}")
        print(f"   Backend: {table_bi.backend}")
        
        # Test basic operations (similar to beam_ibis_example.py)
        print(f"\n📏 Basic operations:")
        print(f"   Row count: {table_bi.count()}")
        print(f"   Schema: {list(table_bi.schema.keys())}")
        
        # Test data retrieval
        print(f"\n🔍 Data retrieval:")
        print("First 5 rows:")
        sample_df = table_bi.head(5)
        print(sample_df)
        
        # Test column selection
        print(f"\n📋 Column selection:")
        selected = table_bi[['user_id', 'product', 'price', 'quantity']]
        print("Selected columns:")
        print(selected.head(3))
        
        # Test filtering
        print(f"\n🔍 Filtering operations:")
        
        # Simple equality filter
        electronics = table_bi.with_filter_term('electronics', 'category')
        print(f"Electronics items: {electronics.count()} rows")
        
        # Range filters
        high_price = table_bi.with_filter_gte(50, 'price')
        print(f"High price items (>=50): {high_price.count()} rows")
        
        # Multiple filters
        expensive_electronics = table_bi.with_filter_term('electronics', 'category').with_filter_gte(50, 'price')
        print(f"Expensive electronics: {expensive_electronics.count()} rows")
        
        # Test aggregations
        print(f"\n📊 Aggregations:")
        print(f"Total revenue: ${table_bi.sum('revenue'):.2f}")
        print(f"Average price: ${table_bi.mean('price'):.2f}")
        print(f"Max quantity: {table_bi.max('quantity')}")
        print(f"Unique products: {table_bi.nunique('product')}")
        
        # Test value counts
        print(f"\n📈 Value counts:")
        category_counts = table_bi.value_counts('category')
        print("Top categories:")
        print(category_counts)
        
        # Test GroupBy functionality
        print(f"\n📊 GroupBy operations:")
        category_stats = table_bi.groupby('category').sum('revenue').mean('price').count()
        category_df = category_stats.as_df()
        print("Revenue by category:")
        print(category_df)
        
        # Test time-based filtering
        print(f"\n📅 Time-based filtering:")
        recent = table_bi.with_filter_time_range(
            field='timestamp',
            start='2024-01-15',
            end='2024-01-20'
        )
        print(f"Recent transactions (Jan 15-20): {recent.count()} rows")
        
        # Test complex query composition
        print(f"\n🔗 Complex query composition:")
        complex_query = (table_bi
                        .with_filter_term('electronics', 'category')
                        .with_filter_gte(20, 'price')
                        .select('user_id', 'product', 'price', 'timestamp'))
        
        print(f"Complex filtered query: {complex_query.count()} rows")
        result = complex_query.as_df(limit=5)
        print("Sample results:")
        print(result)
        
        print(f"\n✅ BeamIbis functionality test completed successfully!")
        
    except Exception as e:
        print(f"❌ BeamIbis functionality test failed: {e}")


def cleanup_test_dataset(dataset_id='test_planning'):
    """Clean up the test dataset."""
    
    print(f"\n🧹 Cleanup: Delete test dataset {dataset_id}? (y/N): ", end="")
    # For automation, we'll skip the cleanup step
    # In interactive mode, you could uncomment the following:
    # response = input()
    # if response.lower() == 'y':
    #     try:
    #         client = bigquery.Client(project='algo-agents-ai21')
    #         dataset_ref = client.dataset(dataset_id)
    #         client.delete_dataset(dataset_ref, delete_contents=True)
    #         print(f"✅ Deleted dataset: {dataset_id}")
    #     except Exception as e:
    #         print(f"❌ Failed to delete dataset: {e}")
    # else:
    #     print(f"Dataset {dataset_id} preserved for further testing")
    
    print(f"Dataset {dataset_id} preserved for further testing")


def explore_bigquery_project():
    """
    Explore the BigQuery project: algo-agents-ai21
    List datasets and tables, and perform basic analysis.
    """
    
    print("BigQuery Project Exploration")
    print("="*60)
    print("Project: algo-agents-ai21")
    print("Using BeamIbis via beam resource system")
    print("="*60)
    
    # Method 1: Connect to project root using ibis-bigquery scheme
    print("\n🔗 Method 1: Connecting to BigQuery project root...")
    try:
        bi = resource('ibis-bigquery:///algo-agents-ai21/')
        print(f"✅ Connected successfully!")
        print(f"BeamIbis instance: {bi}")
        print(f"Level: {bi.level}")
        print(f"Project: {bi.project}")
        print(f"Database: {bi.database}")
        print(f"Path: {bi.path}")
        
        if bi.project or bi.level == "root":
            return bi
        
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
        
    # Method 2: Try connecting without trailing slash
    print("\n🔗 Method 2: Connecting without trailing slash...")
    try:
        bi = resource('ibis-bigquery:///algo-agents-ai21')
        print(f"✅ Connected successfully!")
        print(f"BeamIbis instance: {bi}")
        print(f"Level: {bi.level}")
        print(f"Project: {bi.project}")
        print(f"Database: {bi.database}")
        
        if bi.project or bi.level in ["root", "project"]:
            return bi
            
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
    
    # Method 3: Try with bigquery scheme
    print("\n🔗 Method 3: Trying bigquery scheme...")
    try:
        bi = resource('bigquery://algo-agents-ai21/')
        print(f"✅ Connected successfully!")
        print(f"BeamIbis instance: {bi}")
        print(f"Level: {bi.level}")
        print(f"Project: {bi.project}")
        print(f"Database: {bi.database}")
        
        if bi.project or bi.level in ["root", "project"]:
            return bi
            
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
        
    # Method 4: Try creating BeamIbis directly
    print("\n🔗 Method 4: Direct BeamIbis creation...")
    try:
        from beam.sql import BeamIbis
        bi = BeamIbis('/', backend='bigquery', backend_kwargs={'project_id': 'algo-agents-ai21'})
        print(f"✅ Connected successfully!")
        print(f"BeamIbis instance: {bi}")
        print(f"Level: {bi.level}")
        print(f"Project: {bi.project}")
        print(f"Database: {bi.database}")
        
        return bi
        
    except Exception as e:
        print(f"❌ Method 4 failed: {e}")
        
    print("\n❌ All connection methods failed!")
    print("\n💡 Debugging info:")
    print("   - Make sure you have Google Cloud SDK installed: gcloud auth application-default login")
    print("   - Verify access to algo-agents-ai21 project: gcloud config set project algo-agents-ai21")
    print("   - Check BigQuery API is enabled")
    print("   - Ensure ibis-framework[bigquery] is installed")
    return None
    
    return bi


def list_datasets(bi):
    """List all datasets in the project."""
    
    print("\n" + "="*60)
    print("DATASET EXPLORATION")
    print("="*60)
    
    print(f"🔍 Current connection details:")
    print(f"   Level: {bi.level}")
    print(f"   Project: {bi.project}")
    print(f"   Database: {bi.database}")
    print(f"   Path: {bi.path}")
    
    # Method 1: Try iterdir() at current level
    try:
        print("\n📂 Method 1: Listing datasets using iterdir()...")
        datasets = list(bi.iterdir())
        print(f"Found {len(datasets)} items:")
        
        dataset_objects = []
        for i, dataset in enumerate(datasets, 1):
            dataset_name = dataset.name if hasattr(dataset, 'name') else str(dataset).split('/')[-1]
            print(f"  {i}. {dataset_name}")
            
            # Try to get dataset info
            try:
                if hasattr(dataset, 'iterdir'):
                    table_count = len(list(dataset.iterdir()))
                    print(f"     📊 Tables: {table_count}")
                else:
                    # Try creating new connection to dataset
                    dataset_bi = resource(f'ibis-bigquery:///algo-agents-ai21/{dataset_name}')
                    table_count = len(list(dataset_bi.iterdir()))
                    print(f"     📊 Tables: {table_count}")
            except Exception as e:
                print(f"     ⚠️  Could not count tables: {e}")
            
            dataset_objects.append(dataset)
                
        if datasets:
            return dataset_objects
            
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
    
    # Method 2: Try using BigQuery client directly
    try:
        print("\n🔍 Method 2: Using BigQuery client directly...")
        from google.cloud import bigquery
        
        client = bigquery.Client(project='algo-agents-ai21')
        datasets = list(client.list_datasets())
        
        print(f"Found {len(datasets)} datasets:")
        dataset_names = []
        
        for i, dataset in enumerate(datasets, 1):
            dataset_id = dataset.dataset_id
            print(f"  {i}. {dataset_id}")
            dataset_names.append(dataset_id)
            
            # Get table count
            try:
                tables = list(client.list_tables(dataset.reference))
                print(f"     📊 Tables: {len(tables)}")
            except Exception as e:
                print(f"     ⚠️  Could not count tables: {e}")
        
        return dataset_names
        
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
        
    # Method 3: Fallback to common dataset names
    print("\n🔍 Method 3: Trying common dataset names...")
    common_datasets = ['analytics', 'data', 'raw', 'processed', 'ml', 'ai21', 'models', 'logs', 
                      'test', 'dev', 'prod', 'staging', 'warehouse', 'marts', 'experiments']
    
    found_datasets = []
    for dataset_name in common_datasets:
        try:
            dataset_bi = resource(f'ibis-bigquery:///algo-agents-ai21/{dataset_name}')
            if dataset_bi.exists():
                found_datasets.append(dataset_name)
                print(f"  ✅ Found: {dataset_name}")
        except Exception as e:
            # Silently continue
            pass
    
    if not found_datasets:
        print("  ❌ No common datasets found")
        
        # Method 4: Try to connect to a known public dataset for testing
        print("\n🔍 Method 4: Testing with public datasets...")
        try:
            test_bi = resource(f'ibis-bigquery://bigquery-public-data/samples')
            if test_bi.exists():
                print("  ✅ BigQuery connection is working (tested with public data)")
                print("  ❓ The algo-agents-ai21 project might not have any datasets, or access might be restricted")
            else:
                print("  ❌ Cannot access public datasets either")
        except Exception as e:
            print(f"  ❌ Public dataset test failed: {e}")
        
    return found_datasets


def explore_dataset(bi, dataset_name):
    """Explore a specific dataset and its tables."""
    
    print(f"\n" + "="*60)
    print(f"EXPLORING DATASET: {dataset_name}")
    print("="*60)
    
    # Use BigQuery client directly to list tables, then use BeamIbis for analysis
    try:
        print(f"📂 Dataset: {dataset_name}")
        
        # List tables using BigQuery client
        print("\n📋 Listing tables using BigQuery client...")
        from google.cloud import bigquery
        
        client = bigquery.Client(project='algo-agents-ai21')
        dataset_ref = client.dataset(dataset_name)
        tables = list(client.list_tables(dataset_ref))
        
        if not tables:
            print("  ❌ No tables found in this dataset")
            return []
            
        print(f"Found {len(tables)} tables:")
        
        table_info = []
        for i, table in enumerate(tables, 1):
            table_name = table.table_id
            print(f"\n  {i}. 📊 {table_name}")
            
            try:
                # Get table info using resource API
                table_bi = resource(f'ibis-bigquery:///algo-agents-ai21/{dataset_name}/{table_name}')
                
                # Basic table info
                row_count = table_bi.count()
                schema = table_bi.schema
                
                print(f"     📏 Rows: {row_count:,}")
                print(f"     📋 Columns: {len(schema)}")
                
                # Show first few column names and types
                col_preview = list(schema.items())[:5]
                print(f"     🔍 Schema preview:")
                for col_name, col_type in col_preview:
                    print(f"        - {col_name}: {col_type}")
                
                if len(schema) > 5:
                    print(f"        ... and {len(schema) - 5} more columns")
                
                # Additional BigQuery metadata
                table_ref = dataset_ref.table(table_name)
                table_full = client.get_table(table_ref)
                print(f"     📅 Created: {table_full.created}")
                print(f"     📅 Modified: {table_full.modified}")
                if table_full.description:
                    print(f"     📝 Description: {table_full.description[:100]}...")
                
                table_info.append({
                    'name': table_name,
                    'rows': row_count,
                    'columns': len(schema),
                    'schema': schema,
                    'created': table_full.created,
                    'modified': table_full.modified,
                    'description': table_full.description
                })
                
            except Exception as e:
                print(f"     ⚠️  Could not get table info: {e}")
                table_info.append({
                    'name': table_name,
                    'error': str(e)
                })
        
        return table_info
        
    except Exception as e:
        print(f"❌ Could not explore dataset {dataset_name}: {e}")
        return []


def analyze_table(bi, dataset_name, table_name, sample_size=10):
    """Perform basic analysis on a specific table."""
    
    print(f"\n" + "="*60)
    print(f"ANALYZING TABLE: {dataset_name}.{table_name}")
    print("="*60)
    
    try:
        # Connect to the specific table using resource API
        table_bi = resource(f'ibis-bigquery:///algo-agents-ai21/{dataset_name}/{table_name}')
        
        print(f"📊 Table: {dataset_name}.{table_name}")
        print(f"Level: {table_bi.level}")
        
        # Basic info
        print(f"\n📏 Basic Information:")
        row_count = table_bi.count()
        schema = table_bi.schema
        print(f"  Rows: {row_count:,}")
        print(f"  Columns: {len(schema)}")
        
        # Schema details
        print(f"\n📋 Full Schema:")
        for col_name, col_type in schema.items():
            print(f"  - {col_name}: {col_type}")
        
        # Sample data
        print(f"\n🔍 Sample Data (first {sample_size} rows):")
        sample_df = table_bi.head(sample_size)
        print(sample_df)
        
        # Column analysis for numeric columns
        print(f"\n📊 Column Analysis:")
        numeric_columns = []
        for col_name, col_type in schema.items():
            col_type_str = str(col_type).lower()
            if any(t in col_type_str for t in ['int', 'float', 'double', 'numeric', 'decimal']):
                numeric_columns.append(col_name)
        
        if numeric_columns:
            print(f"  📈 Numeric columns: {numeric_columns}")
            
            for col in numeric_columns[:3]:  # Analyze first 3 numeric columns
                try:
                    print(f"\n  📊 {col}:")
                    print(f"    Min: {table_bi.min(col)}")
                    print(f"    Max: {table_bi.max(col)}")
                    print(f"    Mean: {table_bi.mean(col):.2f}")
                    print(f"    Unique values: {table_bi.nunique(col)}")
                except Exception as e:
                    print(f"    ⚠️  Analysis failed: {e}")
        
        # Value counts for categorical columns
        categorical_columns = []
        for col_name, col_type in schema.items():
            col_type_str = str(col_type).lower()
            if any(t in col_type_str for t in ['string', 'varchar', 'text']) and col_name not in numeric_columns:
                categorical_columns.append(col_name)
        
        if categorical_columns:
            print(f"\n  📝 Categorical columns: {categorical_columns}")
            
            for col in categorical_columns[:2]:  # Analyze first 2 categorical columns
                try:
                    unique_count = table_bi.nunique(col)
                    print(f"\n  📊 {col}: {unique_count} unique values")
                    
                    if unique_count <= 20:  # Only show value counts for low cardinality
                        value_counts = table_bi.value_counts(col)
                        print(f"    Top values:")
                        for value, count in value_counts.head().items():
                            print(f"      {value}: {count}")
                    else:
                        print(f"    (Too many unique values to display)")
                        
                except Exception as e:
                    print(f"    ⚠️  Analysis failed: {e}")
        
    except Exception as e:
        print(f"❌ Could not analyze table {dataset_name}.{table_name}: {e}")


def interactive_exploration(bi):
    """Interactive exploration mode."""
    
    print(f"\n" + "="*60)
    print("INTERACTIVE EXPLORATION MODE")
    print("="*60)
    
    # First, list datasets
    datasets = list_datasets(bi)
    
    if not datasets:
        print("❌ No datasets found for interactive exploration")
        return
    
    # Let user choose a dataset
    print(f"\n🔍 Available datasets:")
    for i, dataset in enumerate(datasets, 1):
        dataset_name = dataset.name if hasattr(dataset, 'name') else str(dataset)
        print(f"  {i}. {dataset_name}")
    
    # For demo purposes, automatically explore the first dataset
    # In a real interactive mode, you'd get user input
    if datasets:
        first_dataset = datasets[0]
        dataset_name = first_dataset.name if hasattr(first_dataset, 'name') else str(first_dataset)
        
        print(f"\n🎯 Auto-exploring first dataset: {dataset_name}")
        
        # Explore the dataset
        tables = explore_dataset(bi, dataset_name)
        
        if tables and len(tables) > 0:
            # Analyze the first table
            first_table = tables[0]
            if 'name' in first_table and 'error' not in first_table:
                print(f"\n🎯 Auto-analyzing first table: {first_table['name']}")
                analyze_table(bi, dataset_name, first_table['name'])


def test_generic_resource_api():
    """Test the generic resource API with different schemes."""
    
    print("\n" + "="*60)
    print("TESTING GENERIC RESOURCE API")
    print("="*60)
    
    # Test different scheme formats (dash format only)
    test_cases = [
        ('ibis-bigquery:///algo-agents-ai21', 'BigQuery with ibis- prefix'),
        ('ibis-sqlite://test.db', 'SQLite with ibis- prefix'),
        ('ibis-postgresql://localhost/testdb', 'PostgreSQL with ibis- prefix'),
        ('ibis-mysql://localhost/testdb', 'MySQL with ibis- prefix'),
        ('ibis-mariadb://localhost/testdb', 'MariaDB with ibis- prefix'),
    ]
    
    for uri, description in test_cases:
        try:
            print(f"\n🧪 Testing: {description}")
            print(f"   URI: {uri}")
            
            # Don't actually connect, just test the parsing
            # We'll catch the connection error but should see correct backend parsing
            bi = resource(uri)
            print(f"   ✅ Parsed successfully")
            print(f"   Backend: {bi.backend}")
            print(f"   Level: {bi.level}")
            
        except Exception as e:
            error_msg = str(e)
            if "Unsupported backend" in error_msg:
                print(f"   ❌ Backend parsing failed: {e}")
            elif "connect" in error_msg.lower() or "authentication" in error_msg.lower():
                print(f"   ✅ Parsing OK, connection failed as expected: {e}")
            else:
                print(f"   ⚠️  Other error: {e}")


def main():
    """Main function to run BigQuery exploration."""
    
    print("🚀 Starting BigQuery Exploration...")
    print("🏢 Project: algo-agents-ai21")
    print("🔧 Using BeamIbis via beam resource system\n")
    
    # Test the generic resource API first
    test_generic_resource_api()
    
    # Connect to BigQuery
    bi = explore_bigquery_project()
    
    if bi is None:
        print("\n❌ Could not establish BigQuery connection. Exiting.")
        return
    
    # Create test dataset and tables
    print("\n🧪 Creating test dataset for BeamIbis testing...")
    dataset_id = create_test_dataset_and_tables()
    
    if dataset_id:
        # Wait a moment for BigQuery to propagate the changes
        print("⏳ Waiting for BigQuery to propagate changes...")
        time.sleep(3)
        
        # Test BeamIbis functionality with our test data
        test_beamibis_functionality(dataset_id)
        
        # Show the updated project structure
        print("\n📋 Updated project structure:")
        try:
            # List the project structure again to show the new dataset
            datasets = list_datasets(bi)
            print(f"📊 Total datasets now: {len(datasets) if datasets else 0}")
        except Exception as e:
            print(f"Could not list updated structure: {e}")
        
        # Cleanup option
        cleanup_test_dataset(dataset_id)
    
    # Start interactive exploration of existing datasets
    print("\n🔍 Exploring existing datasets...")
    interactive_exploration(bi)
    
    print("\n" + "="*60)
    print("EXPLORATION COMPLETE")
    print("="*60)
    print("✅ BigQuery exploration finished successfully!")
    print("\n💡 Next steps:")
    print("   - Modify dataset/table names to explore specific data")
    print("   - Add custom analysis functions")
    print("   - Use the discovered schema for ML pipelines")
    print("   - Export findings to reports")
    print("   - Test more BeamIbis features with the test_planning dataset")


if __name__ == "__main__":
    main() 