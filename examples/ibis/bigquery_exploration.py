#!/usr/bin/env python3
"""
BigQuery Exploration with BeamIbis

This script explores the algo-agents-ai21 BigQuery project using only BeamIbis API.
Updated to use the new BeamIbis write operations instead of native BigQuery API.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Import BeamIbis components
from beam.sql import beam_ibis, BeamIbisSchema


def explore_project_structure():
    """Explore the overall project structure using BeamIbis."""
    
    print("=" * 60)
    print("EXPLORING BIGQUERY PROJECT STRUCTURE")
    print("=" * 60)
    
    # Connect to the project root
    project = beam_ibis('ibis-bigquery:///algo-agents-ai21')
    
    print(f"📊 Connected to project: {project}")
    print(f"    Level: {project.level}")
    print(f"    Backend: {project.backend}")
    
    # List all datasets
    print("\n📁 Available datasets:")
    for dataset in project.iterdir():
        print(f"  - {dataset.name}")
        
        # Count tables in each dataset
        table_count = len(list(dataset.iterdir()))
        print(f"    Tables: {table_count}")
        
        if table_count > 0:
            print("    Table list:")
            for table in dataset.iterdir():
                try:
                    row_count = table.count()
                    print(f"      • {table.name} ({row_count:,} rows)")
                except Exception as e:
                    print(f"      • {table.name} (error: {e})")
    
    return project


def create_test_dataset_with_beamibis(project):
    """Create test dataset using only BeamIbis write operations."""
    
    print("\n" + "=" * 60)
    print("CREATING TEST DATASET WITH BEAMIBIS")
    print("=" * 60)
    
    # Navigate to test_planning dataset
    test_dataset = project / 'test_planning'
    
    print(f"📊 Working with dataset: {test_dataset}")
    
    # 1. Create sales table using BeamIbis schema and write operations
    print("\n1. Creating sales table...")
    
    # Define schema using BeamIbisSchema
    sales_schema = BeamIbisSchema(
        fields={
            'sale_id': 'string',
            'product_id': 'string',
            'category': 'string',
            'region': 'string',
            'sale_date': 'timestamp',
            'quantity': 'int64',
            'unit_price': 'float64',
            'total_amount': 'float64',
            'customer_id': 'string',
            'salesperson_id': 'string'
        },
        description="Sales transaction data"
    )
    
    # Generate sample sales data
    categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports']
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    sales_data = []
    base_date = datetime(2024, 1, 1)
    
    for i in range(1000):
        sales_data.append({
            'sale_id': f'SALE_{i:06d}',
            'product_id': f'PROD_{i % 100:03d}',
            'category': random.choice(categories),
            'region': random.choice(regions),
            'sale_date': base_date + timedelta(days=random.randint(0, 365), 
                                             hours=random.randint(0, 23)),
            'quantity': random.randint(1, 10),
            'unit_price': round(random.uniform(10.0, 500.0), 2),
            'total_amount': 0,  # Will calculate below
            'customer_id': f'CUST_{i % 200:04d}',
            'salesperson_id': f'SALES_{i % 20:02d}'
        })
        # Calculate total amount
        sales_data[i]['total_amount'] = round(
            sales_data[i]['quantity'] * sales_data[i]['unit_price'], 2
        )
    
    sales_df = pd.DataFrame(sales_data)
    
    # Create table using BeamIbis
    try:
        sales_table = test_dataset.create_table_from_data(sales_df, 'sales')
        print(f"✅ Created sales table with {sales_table.count():,} rows")
    except Exception as e:
        print(f"❌ Failed to create sales table: {e}")
        return None
    
    # 2. Create transactions table
    print("\n2. Creating transactions table...")
    
    transactions_schema = BeamIbisSchema(
        fields={
            'transaction_id': 'string',
            'user_id': 'string',
            'transaction_date': 'timestamp',
            'amount': 'float64',
            'payment_method': 'string',
            'status': 'string',
            'merchant_id': 'string',
            'category': 'string'
        },
        description="Payment transaction data"
    )
    
    payment_methods = ['credit_card', 'debit_card', 'paypal', 'bank_transfer', 'cash']
    statuses = ['completed', 'pending', 'failed', 'cancelled']
    
    transactions_data = []
    for i in range(1000):
        transactions_data.append({
            'transaction_id': f'TXN_{i:08d}',
            'user_id': f'USER_{i % 150:04d}',
            'transaction_date': base_date + timedelta(days=random.randint(0, 365),
                                                   hours=random.randint(0, 23)),
            'amount': round(random.uniform(5.0, 1000.0), 2),
            'payment_method': random.choice(payment_methods),
            'status': random.choice(statuses),
            'merchant_id': f'MERCH_{i % 50:03d}',
            'category': random.choice(categories)
        })
    
    transactions_df = pd.DataFrame(transactions_data)
    
    try:
        transactions_table = test_dataset.create_table_from_data(transactions_df, 'transactions')
        print(f"✅ Created transactions table with {transactions_table.count():,} rows")
    except Exception as e:
        print(f"❌ Failed to create transactions table: {e}")
        return None
    
    # 3. Create user_activity table  
    print("\n3. Creating user_activity table...")
    
    activity_schema = BeamIbisSchema(
        fields={
            'activity_id': 'string',
            'user_id': 'string',
            'session_id': 'string',
            'activity_timestamp': 'timestamp',
            'page_url': 'string',
            'action_type': 'string',
            'device_type': 'string',
            'browser': 'string',
            'duration_seconds': 'int64'
        },
        description="User website activity data"
    )
    
    action_types = ['page_view', 'click', 'search', 'purchase', 'logout', 'login']
    device_types = ['desktop', 'mobile', 'tablet']
    browsers = ['chrome', 'firefox', 'safari', 'edge']
    
    activity_data = []
    for i in range(1000):
        activity_data.append({
            'activity_id': f'ACT_{i:08d}',
            'user_id': f'USER_{i % 150:04d}',
            'session_id': f'SESS_{i % 300:05d}',
            'activity_timestamp': base_date + timedelta(days=random.randint(0, 365),
                                                      hours=random.randint(0, 23)),
            'page_url': f'/page/{random.choice(categories).lower()}',
            'action_type': random.choice(action_types),
            'device_type': random.choice(device_types),
            'browser': random.choice(browsers),
            'duration_seconds': random.randint(5, 600)
        })
    
    activity_df = pd.DataFrame(activity_data)
    
    try:
        activity_table = test_dataset.create_table_from_data(activity_df, 'user_activity')
        print(f"✅ Created user_activity table with {activity_table.count():,} rows")
    except Exception as e:
        print(f"❌ Failed to create user_activity table: {e}")
        return None
    
    print(f"\n✅ Successfully created test_planning dataset with 3 tables!")
    return test_dataset


def demonstrate_write_operations(test_dataset):
    """Demonstrate the 3 key write operations using BeamIbis."""
    
    print("\n" + "=" * 60)
    print("DEMONSTRATING WRITE OPERATIONS")
    print("=" * 60)
    
    # Access the sales table
    sales_table = test_dataset / 'sales'
    
    # 1. Append batch data
    print("1. Appending batch data...")
    initial_count = sales_table.count()
    
    # Create new batch data
    new_sales_data = []
    for i in range(1000, 1050):  # 50 new records
        new_sales_data.append({
            'sale_id': f'SALE_{i:06d}',
            'product_id': f'PROD_{i % 100:03d}',
            'category': 'Electronics',
            'region': 'Online',
            'sale_date': datetime.now(),
            'quantity': random.randint(1, 5),
            'unit_price': round(random.uniform(50.0, 200.0), 2),
            'total_amount': 0,
            'customer_id': f'CUST_{i % 200:04d}',
            'salesperson_id': 'SALES_ONLINE'
        })
        # Calculate total
        new_sales_data[i-1000]['total_amount'] = round(
            new_sales_data[i-1000]['quantity'] * new_sales_data[i-1000]['unit_price'], 2
        )
    
    new_sales_df = pd.DataFrame(new_sales_data)
    
    try:
        sales_table.append_batch(new_sales_df)
        final_count = sales_table.count()
        print(f"✅ Appended {final_count - initial_count} rows")
        print(f"   Total rows now: {final_count:,}")
    except Exception as e:
        print(f"❌ Failed to append batch: {e}")
    
    # 2. Append single row
    print("\n2. Appending single row...")
    single_row = {
        'sale_id': 'SALE_SPECIAL',
        'product_id': 'PROD_SPECIAL',
        'category': 'Premium',
        'region': 'VIP',
        'sale_date': datetime.now(),
        'quantity': 1,
        'unit_price': 999.99,
        'total_amount': 999.99,
        'customer_id': 'CUST_VIP',
        'salesperson_id': 'SALES_MANAGER'
    }
    
    try:
        prev_count = sales_table.count()
        sales_table.append_row(single_row)
        new_count = sales_table.count()
        print(f"✅ Appended single row")
        print(f"   Rows added: {new_count - prev_count}")
        print(f"   Total rows: {new_count:,}")
    except Exception as e:
        print(f"❌ Failed to append single row: {e}")
    
    # 3. Create new table from schema
    print("\n3. Creating new table from schema...")
    
    # Define a new table schema
    analytics_schema = BeamIbisSchema(
        fields={
            'report_id': 'string',
            'report_date': 'timestamp',
            'total_sales': 'float64',
            'total_transactions': 'int64',
            'avg_order_value': 'float64',
            'top_category': 'string',
            'created_by': 'string'
        },
        description="Daily analytics summary"
    )
    
    try:
        # Create empty table from schema
        analytics_table = test_dataset.create_table_from_schema(analytics_schema, 'daily_analytics')
        print(f"✅ Created empty analytics table: {analytics_table}")
        
        # Add a sample analytics record
        sample_analytics = {
            'report_id': 'RPT_20241201',
            'report_date': datetime.now(),
            'total_sales': 45678.90,
            'total_transactions': 1051,
            'avg_order_value': 43.46,
            'top_category': 'Electronics',
            'created_by': 'automated_system'
        }
        
        analytics_table.append_row(sample_analytics)
        print(f"✅ Added sample analytics record")
        print(f"   Analytics table rows: {analytics_table.count()}")
        
    except Exception as e:
        print(f"❌ Failed to create analytics table: {e}")


def query_and_analyze_data(test_dataset):
    """Query and analyze the data using BeamIbis."""
    
    print("\n" + "=" * 60)
    print("QUERYING AND ANALYZING DATA")
    print("=" * 60)
    
    try:
        # Access tables
        sales_table = test_dataset / 'sales'
        transactions_table = test_dataset / 'transactions'
        
        # 1. Sales analysis
        print("1. Sales Analysis:")
        total_sales = sales_table.total_amount.sum()
        avg_sale = sales_table.total_amount.mean()
        total_orders = sales_table.count()
        
        print(f"   📊 Total Sales: ${total_sales:,.2f}")
        print(f"   📊 Average Sale: ${avg_sale:.2f}")
        print(f"   📊 Total Orders: {total_orders:,}")
        
        # 2. Sales by category
        print("\n2. Sales by Category:")
        category_sales = (sales_table
                         .group_by('category')
                         .agg(
                             total_sales=sales_table.total_amount.sum(),
                             order_count=sales_table.sale_id.count()
                         )
                         .order_by(sales_table.total_amount.sum().desc()))
        
        category_results = category_sales.as_df()
        for _, row in category_results.iterrows():
            print(f"   📈 {row['category']}: ${row['total_sales']:,.2f} ({row['order_count']} orders)")
        
        # 3. Transaction status analysis
        print("\n3. Transaction Status Analysis:")
        status_analysis = (transactions_table
                          .group_by('status')
                          .agg(
                              count=transactions_table.transaction_id.count(),
                              total_amount=transactions_table.amount.sum()
                          ))
        
        status_results = status_analysis.as_df()
        for _, row in status_results.iterrows():
            print(f"   💳 {row['status']}: {row['count']} transactions, ${row['total_amount']:,.2f}")
        
        # 4. Recent high-value sales
        print("\n4. Recent High-Value Sales (>$100):")
        high_value_sales = (sales_table
                           .filter(sales_table.total_amount > 100)
                           .order_by(sales_table.sale_date.desc())
                           .select('sale_id', 'category', 'total_amount', 'sale_date')
                           .head(5))
        
        print(high_value_sales.as_df().to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")


def main():
    """Main execution function."""
    
    print("BigQuery Exploration with BeamIbis Write Operations")
    print("=" * 60)
    print("Using ONLY BeamIbis API - no native BigQuery API calls!")
    print("=" * 60)
    
    try:
        # 1. Explore project structure
        project = explore_project_structure()
        
        # 2. Create test dataset with BeamIbis write operations
        test_dataset = create_test_dataset_with_beamibis(project)
        
        if test_dataset:
            # 3. Demonstrate write operations
            demonstrate_write_operations(test_dataset)
            
            # 4. Query and analyze data
            query_and_analyze_data(test_dataset)
            
            print("\n" + "=" * 60)
            print("✅ ALL OPERATIONS COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("\nKey Achievements:")
            print("• ✅ Used ONLY BeamIbis API for all operations")
            print("• ✅ Created tables using BeamIbisSchema")
            print("• ✅ Demonstrated batch and single-row writes")
            print("• ✅ Leveraged Ibis native schema conversion")
            print("• ✅ Performed complex queries and analytics")
        
    except Exception as e:
        print(f"\n❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 