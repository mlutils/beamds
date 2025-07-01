#!/usr/bin/env python3
"""
BeamIbis Example - Comprehensive demonstration of the BeamIbis class

This example shows how to use BeamIbis, which provides a unified interface
similar to pathlib+pandas for all Ibis clients (BigQuery, SQLite, PostgreSQL, etc.)

The API is designed to be similar to BeamElastic but adapted for SQL databases.
"""

import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import tempfile
import os

# Import the BeamIbis classes
from beam.sql import BeamIbis, beam_ibis


def create_sample_data():
    """Create sample data for demonstration."""
    
    # Create sample data similar to what you might have in Elasticsearch
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


def setup_sqlite_database():
    """Set up a SQLite database with sample data."""
    
    # Create a temporary database file
    db_path = tempfile.mktemp(suffix='.db')
    
    # Create sample data
    df = create_sample_data()
    
    # Create SQLite connection and write data
    conn = sqlite3.connect(db_path)
    df.to_sql('sales', conn, index=False, if_exists='replace')
    df.to_sql('transactions', conn, index=False, if_exists='replace')  # Duplicate for demo
    conn.close()
    
    print(f"Created SQLite database: {db_path}")
    print(f"Sample data shape: {df.shape}")
    print(f"Sample data columns: {list(df.columns)}")
    
    return db_path


def demonstrate_basic_usage(db_path):
    """Demonstrate basic BeamIbis usage."""
    
    print("\n" + "="*60)
    print("BASIC USAGE DEMONSTRATION")
    print("="*60)
    
    # Create BeamIbis instance using different methods
    
    # Method 1: Direct instantiation
    beam_db = BeamIbis(f"/{db_path}/sales", backend='sqlite')
    print(f"BeamIbis instance: {beam_db}")
    print(f"Level: {beam_db.level}")
    print(f"Table name: {beam_db.table_name}")
    print(f"Database: {beam_db.database}")
    
    # Method 2: Using beam_ibis function with URL
    beam_db2 = beam_ibis(f"sqlite://{db_path}/sales")
    print(f"\nUsing beam_ibis function: {beam_db2}")
    
    # Basic operations
    print(f"\nTable exists: {beam_db.exists()}")
    print(f"Row count: {beam_db.count()}")
    print(f"Schema: {list(beam_db.schema.keys())}")


def demonstrate_querying(db_path):
    """Demonstrate querying capabilities."""
    
    print("\n" + "="*60)
    print("QUERYING DEMONSTRATION")
    print("="*60)
    
    beam_db = BeamIbis(f"/{db_path}/sales", backend='sqlite')
    
    # Basic data retrieval
    print("First 5 rows:")
    print(beam_db.head())
    
    # Column selection
    print("\nSelecting specific columns:")
    selected = beam_db[['user_id', 'product', 'price', 'quantity']]
    print(selected.head())
    
    # Filtering (similar to BeamElastic)
    print("\nFiltering examples:")
    
    # Simple equality filter
    electronics = beam_db.with_filter_term('electronics', 'category')
    print(f"Electronics items: {electronics.count()} rows")
    
    # Range filters
    high_price = beam_db.with_filter_gte(50, 'price')
    print(f"High price items (>=50): {high_price.count()} rows")
    
    # Multiple filters (query composition)
    expensive_electronics = beam_db.with_filter_term('electronics', 'category') & beam_db.with_filter_gte(50, 'price')
    print(f"Expensive electronics: {expensive_electronics.count()} rows")
    
    # Time range filtering
    recent = beam_db.with_filter_time_range(
        field='timestamp',
        start='2024-01-15',
        end='2024-01-20'
    )
    print(f"Recent transactions: {recent.count()} rows")
    
    # Show sample of filtered data
    print("\nSample of expensive electronics:")
    print(expensive_electronics.head(3))


def demonstrate_aggregations(db_path):
    """Demonstrate aggregation capabilities."""
    
    print("\n" + "="*60)
    print("AGGREGATIONS DEMONSTRATION")
    print("="*60)
    
    beam_db = BeamIbis(f"/{db_path}/sales", backend='sqlite')
    
    # Basic aggregations
    print("Basic aggregations:")
    print(f"Total revenue: ${beam_db.sum('revenue'):.2f}")
    print(f"Average price: ${beam_db.mean('price'):.2f}")
    print(f"Max quantity: {beam_db.max('quantity')}")
    print(f"Unique products: {beam_db.nunique('product')}")
    
    # Value counts (similar to BeamElastic)
    print("\nTop categories by count:")
    category_counts = beam_db.value_counts('category')
    print(category_counts)
    
    print("\nTop countries by count:")
    country_counts = beam_db.value_counts('country')
    print(country_counts.head())


def demonstrate_groupby(db_path):
    """Demonstrate GroupBy functionality."""
    
    print("\n" + "="*60)
    print("GROUPBY DEMONSTRATION")
    print("="*60)
    
    beam_db = BeamIbis(f"/{db_path}/sales", backend='sqlite')
    
    # Simple groupby
    print("Revenue by category:")
    category_stats = beam_db.groupby('category').sum('revenue').mean('price').count()
    print(category_stats.as_df())
    
    # Multiple groupby fields
    print("\nRevenue by category and country:")
    multi_group = beam_db.groupby(['category', 'country']).sum('revenue').count()
    multi_df = multi_group.as_df()
    print(multi_df.head(10))
    
    # Complex aggregations
    print("\nComplex aggregations by product:")
    product_stats = (beam_db.groupby('product')
                     .sum('revenue')
                     .mean('price')
                     .max('quantity')
                     .nunique('user_id'))
    
    product_df = product_stats.as_df()
    print(product_df.head(10))
    
    # Using agg method for multiple aggregations
    print("\nUsing .agg() method:")
    agg_stats = beam_db.groupby('category').agg({
        'revenue': ['sum', 'mean'],
        'quantity': ['sum', 'max'],
        'user_id': 'nunique'
    })
    print(agg_stats.as_df())


def demonstrate_output_formats(db_path):
    """Demonstrate different output formats."""
    
    print("\n" + "="*60)
    print("OUTPUT FORMATS DEMONSTRATION")
    print("="*60)
    
    beam_db = BeamIbis(f"/{db_path}/sales", backend='sqlite')
    
    # Limit data for demo
    sample_query = beam_db.with_filter_term('electronics', 'category')
    
    # Pandas DataFrame (default)
    print("As Pandas DataFrame:")
    df = sample_query.as_df(limit=5)
    print(df)
    print(f"Type: {type(df)}")
    
    # Dictionary format
    print("\nAs Dictionary:")
    dict_data = sample_query.as_dict(limit=3)
    print(f"First record: {dict_data[0]}")
    print(f"Type: {type(dict_data)}")
    
    # Polars DataFrame (if available)
    try:
        print("\nAs Polars DataFrame:")
        pl_df = sample_query.as_pl(limit=5)
        print(pl_df)
        print(f"Type: {type(pl_df)}")
    except ImportError:
        print("\nPolars not available - install with: pip install polars")
    
    # SQL representation
    print("\nSQL Query:")
    try:
        sql = sample_query.sql()
        print(sql)
    except Exception as e:
        print(f"SQL generation error: {e}")


def demonstrate_lazy_execution(db_path):
    """Demonstrate lazy execution."""
    
    print("\n" + "="*60)
    print("LAZY EXECUTION DEMONSTRATION")
    print("="*60)
    
    beam_db = BeamIbis(f"/{db_path}/sales", backend='sqlite')
    
    # Build complex query without execution
    print("Building complex query (no execution yet):")
    
    complex_query = (beam_db
                     .with_filter_term('electronics', 'category')
                     .with_filter_gte(20, 'price')
                     .order_by('timestamp')
                     .select(['user_id', 'product', 'price', 'timestamp']))
    
    print(f"Complex query object: {complex_query}")
    print(f"Level: {complex_query.level}")
    
    # Only executes when we call as_df(), as_dict(), etc.
    print("\nExecuting query now:")
    result = complex_query.as_df(limit=10)
    print(result)
    
    # Show that each operation returns a new object (immutable)
    original_count = beam_db.count()
    filtered_count = beam_db.with_filter_term('electronics', 'category').count()
    
    print(f"\nOriginal count: {original_count}")
    print(f"Filtered count: {filtered_count}")
    print(f"Original still unchanged: {beam_db.count()}")


def demonstrate_path_like_interface(db_path):
    """Demonstrate path-like interface."""
    
    print("\n" + "="*60)
    print("PATH-LIKE INTERFACE DEMONSTRATION")
    print("="*60)
    
    # Navigate like filesystem
    root = BeamIbis("/", backend='sqlite', database=db_path)
    print(f"Root level: {root}")
    print(f"Level: {root.level}")
    
    # Navigate to table
    sales_table = BeamIbis(f"/{db_path}/sales", backend='sqlite')
    print(f"Table level: {sales_table}")
    print(f"Level: {sales_table.level}")
    
    # List tables in database
    try:
        dataset = BeamIbis(f"/{db_path}", backend='sqlite')
        print(f"Dataset level: {dataset}")
        print("Tables in database:")
        for table in dataset.iterdir():
            print(f"  - {table}")
    except Exception as e:
        print(f"Directory listing error: {e}")


def demonstrate_comparison_operators(db_path):
    """Demonstrate comparison operators like BeamElastic."""
    
    print("\n" + "="*60)
    print("COMPARISON OPERATORS DEMONSTRATION")
    print("="*60)
    
    beam_db = BeamIbis(f"/{db_path}/sales", backend='sqlite')
    
    # Note: These work when you have a single column selected
    price_column = beam_db['price']
    
    print("Using comparison operators (requires single column):")
    try:
        high_prices = price_column >= 50
        print(f"Items with price >= 50: {high_prices.count()} rows")
        
        moderate_prices = price_column < 30
        print(f"Items with price < 30: {moderate_prices.count()} rows")
    except Exception as e:
        print(f"Comparison operator error: {e}")
        print("Note: Comparison operators work best with single column selection")


def main():
    """Main demonstration function."""
    
    print("BeamIbis Comprehensive Demonstration")
    print("="*60)
    print("This demonstrates a unified interface similar to pathlib+pandas")
    print("for all Ibis clients (BigQuery, SQLite, PostgreSQL, etc.)")
    print("The API is designed to be similar to BeamElastic but for SQL databases.")
    
    # Set up demo database
    db_path = setup_sqlite_database()
    
    try:
        # Run all demonstrations
        demonstrate_basic_usage(db_path)
        demonstrate_querying(db_path)
        demonstrate_aggregations(db_path)
        demonstrate_groupby(db_path)
        demonstrate_output_formats(db_path)
        demonstrate_lazy_execution(db_path)
        demonstrate_path_like_interface(db_path)
        demonstrate_comparison_operators(db_path)
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✓ Lazy query execution")
        print("✓ Path-like interface for navigation")
        print("✓ Query composition with & and | operators")
        print("✓ Multiple filtering methods")
        print("✓ Comprehensive aggregation support")
        print("✓ GroupBy operations similar to pandas")
        print("✓ Multiple output formats (pandas, polars, cudf, dict)")
        print("✓ Time range filtering")
        print("✓ Immutable query objects")
        print("✓ Multiple backend support")
        
        print("\nSupported Backends:")
        print("- SQLite")
        print("- BigQuery")
        print("- PostgreSQL")
        print("- MySQL")
        print("- DuckDB")
        print("- And more via Ibis")
        
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)
            print(f"\nCleaned up database: {db_path}")


if __name__ == "__main__":
    main() 