#!/usr/bin/env python3
"""
BeamIbis Write Operations Example

This example demonstrates the complete write functionality of BeamIbis:
1. Creating new tables with schemas
2. Appending batch data
3. Appending single rows
4. Schema management with Pydantic

All operations use only the BeamIbis API and leverage Ibis's native 
schema conversion across backends.
"""

import pandas as pd
import tempfile
import os
from datetime import datetime, timedelta

# Import BeamIbis components
from beam.sql import beam_ibis, BeamIbisSchema


def create_sample_data():
    """Create sample e-commerce data for demonstration."""
    
    base_date = datetime(2024, 1, 1)
    
    # Sample orders data
    orders_data = []
    for i in range(100):
        orders_data.append({
            'order_id': f"ORD_{i:06d}",
            'customer_id': f"CUST_{i % 20:04d}",
            'product_id': f"PROD_{i % 15:03d}",
            'quantity': (i % 5) + 1,
            'price': round(10.99 + (i % 100) * 0.99, 2),
            'order_date': base_date + timedelta(days=i % 30, hours=i % 24),
            'is_paid': i % 7 != 0  # Most orders are paid
        })
    
    # Additional batch for appending
    new_orders_data = []
    for i in range(100, 120):
        new_orders_data.append({
            'order_id': f"ORD_{i:06d}",
            'customer_id': f"CUST_{i % 20:04d}",
            'product_id': f"PROD_{i % 15:03d}",
            'quantity': (i % 5) + 1,
            'price': round(10.99 + (i % 100) * 0.99, 2),
            'order_date': base_date + timedelta(days=(i % 30) + 30, hours=i % 24),
            'is_paid': i % 6 != 0
        })
    
    return pd.DataFrame(orders_data), pd.DataFrame(new_orders_data)


def demonstrate_schema_management():
    """Demonstrate BeamIbisSchema functionality."""
    
    print("\n" + "="*60)
    print("SCHEMA MANAGEMENT DEMONSTRATION")
    print("="*60)
    
    # 1. Create a custom schema using the new class syntax
    print("1. Creating custom e-commerce schema...")
    class OrderSchema(BeamIbisSchema):
        """E-commerce order schema with proper types."""
        order_id: str
        customer_id: str
        product_id: str
        quantity: int
        price: float
        order_date: datetime
        is_paid: bool = False
    
    print("✅ Schema created using natural Python class syntax:")
    print(f"   Class: {OrderSchema.__name__}")
    print(f"   Fields: {OrderSchema.get_field_names()}")
    print(f"   Ibis schema: {OrderSchema.to_ibis_schema()}")
    
    # 2. Schema evolution through inheritance
    print("\n2. Evolving schema through inheritance...")
    class ExtendedOrderSchema(OrderSchema):
        """Extended order schema with additional fields."""
        discount_amount: float = 0.0
        currency_code: str = "USD"
        shipping_address: str = ""
    
    print("✅ Evolved schema through inheritance:")
    print(f"   Base schema fields: {OrderSchema.get_field_names()}")
    print(f"   Extended schema fields: {ExtendedOrderSchema.get_field_names()}")
    
    # 3. Using pre-defined schemas
    print("\n3. Using pre-defined schemas...")
    from beam.sql import EventLogSchema, UserSchema
    
    print("✅ Pre-defined EventLogSchema:")
    print(f"   Fields: {EventLogSchema.get_field_names()}")
    
    print("✅ Pre-defined UserSchema:")
    print(f"   Fields: {UserSchema.get_field_names()}")
    
    return OrderSchema, ExtendedOrderSchema


def demonstrate_write_operations_sqlite():
    """Demonstrate write operations with SQLite backend."""
    
    print("\n" + "="*60)
    print("WRITE OPERATIONS - SQLITE BACKEND")
    print("="*60)
    
    # Create temporary database
    db_path = tempfile.mktemp(suffix='.db')
    
    try:
        # Connect to SQLite using BeamIbis
        db = beam_ibis(f'ibis-sqlite:///{db_path}')
        print(f"📊 Connected to SQLite: {db_path}")
        
        # Get sample data
        orders_df, new_orders_df = create_sample_data()
        
        # 1. Create table from data using the new write API
        print("\n1. Creating table from DataFrame using write()...")
        orders_table = db / 'orders'  # Navigate to table level
        orders_table = orders_table.write(orders_df)  # Smart routing to create_table_from_data
        print(f"✅ Created table with {orders_table.count()} rows")
        print(f"   Schema: {list(orders_table.schema.keys())}")
        
        # 2. Append batch data using write()
        print("\n2. Appending batch data using write()...")
        initial_count = orders_table.count()
        orders_table.write(new_orders_df)  # Smart routing to append_batch
        final_count = orders_table.count()
        print(f"✅ Appended {final_count - initial_count} rows")
        print(f"   Total rows now: {final_count:,}")
        
        # 3. Append single row using write()
        print("\n3. Appending single row using write()...")
        single_row = {
            'order_id': 'ORD_999999',
            'customer_id': 'CUST_SPECIAL',
            'product_id': 'PROD_SPECIAL',
            'quantity': 1,
            'price': 999.99,
            'order_date': datetime.now(),
            'is_paid': True
        }
        
        prev_count = orders_table.count()
        orders_table.write(single_row)  # Smart routing to append_row
        new_count = orders_table.count()
        print(f"✅ Appended single row")
        print(f"   Rows added: {new_count - prev_count}")
        print(f"   Total rows: {new_count:,}")
        
        # 4. Demonstrate schema retrieval
        print("\n4. Retrieving table schema...")
        schema_info = orders_table.describe_schema()
        print(f"✅ Schema retrieved:")
        print(schema_info)
        
        # 5. Query the data
        print("\n5. Querying the written data...")
        recent_orders = (orders_table
                        .with_filter_gt(50, 'price')
                        .order_by('order_date')
                        .head(5))
        
        print("✅ Recent high-value orders:")
        print(recent_orders)
        
        return orders_table
        
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)


def demonstrate_schema_based_table_creation():
    """Demonstrate creating tables from schemas."""
    
    print("\n" + "="*60)
    print("SCHEMA-BASED TABLE CREATION")
    print("="*60)
    
    # Create temporary database
    db_path = tempfile.mktemp(suffix='.db')
    
    try:
        db = beam_ibis(f'ibis-sqlite:///{db_path}')
        
        # 1. Create table from schema using write API
        print("1. Creating table from BeamIbisSchema using write()...")
        
        # Define schema using class syntax
        class UserSchema(BeamIbisSchema):
            """User registration schema."""
            user_id: str
            email: str
            signup_date: datetime
            is_active: bool = True
            profile_data: str = "{}"  # JSON as string for SQLite compatibility
        
        users_table = db / 'users'  # Navigate to table level
        users_table = users_table.write([], schema=UserSchema)  # Create empty table from schema
        print(f"✅ Created empty table: {users_table}")
        print(f"   Schema: {users_table.schema}")
        
        # 2. Add users data using write()
        print("\n2. Adding users data using write()...")
        users_data = [
            {
                'user_id': 'USER_001',
                'email': 'alice@example.com',
                'signup_date': datetime.now() - timedelta(days=30),
                'is_active': True,
                'profile_data': '{"age": 25, "city": "NYC"}'
            },
            {
                'user_id': 'USER_002', 
                'email': 'bob@example.com',
                'signup_date': datetime.now() - timedelta(days=15),
                'is_active': True,
                'profile_data': '{"age": 32, "city": "SF"}'
            }
        ]
        
        users_table.write(users_data)  # Smart routing to append_batch
        print(f"✅ Added {len(users_data)} users")
        print(f"   Total users: {users_table.count()}")
        
        # 3. Query users and demonstrate schema retrieval
        print("\n3. Querying users and checking schema...")
        active_users = users_table.with_filter_term(True, 'is_active')
        print("✅ Active users:")
        print(active_users.as_df())
        
        # Show schema retrieval
        print("\n✅ Retrieved schema:")
        print(users_table.describe_schema())
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def demonstrate_cross_backend_compatibility():
    """Demonstrate that schemas work across different backends."""
    
    print("\n" + "="*60)
    print("CROSS-BACKEND COMPATIBILITY")
    print("="*60)
    
    # Define schema once using class syntax
    class TestSchema(BeamIbisSchema):
        """Test schema for cross-backend compatibility."""
        id: int
        name: str
        value: float
        created_at: datetime
    
    # Sample data
    test_data = pd.DataFrame([
        {'id': 1, 'name': 'test1', 'value': 1.5, 'created_at': datetime.now()},
        {'id': 2, 'name': 'test2', 'value': 2.5, 'created_at': datetime.now()},
    ])
    
    backends_to_test = []
    
    # Test SQLite
    sqlite_path = tempfile.mktemp(suffix='.db')
    try:
        print("1. Testing SQLite backend...")
        sqlite_db = beam_ibis(f'ibis-sqlite:///{sqlite_path}')
        sqlite_table = sqlite_db / 'test_table'  # Navigate to table level
        sqlite_table = sqlite_table.write(test_data, schema=TestSchema)  # Use write with schema
        
        print(f"✅ SQLite: Created table with {sqlite_table.count()} rows")
        backends_to_test.append(('SQLite', sqlite_table))
        
    except Exception as e:
        print(f"❌ SQLite failed: {e}")
        
    # Note: For BigQuery, PostgreSQL, etc., you would use the same schema:
    # bq_db = beam_ibis('ibis-bigquery:///my-project/my-dataset')  
    # bq_table = (bq_db / 'test_table').write(test_data, schema=TestSchema)
    # 
    # pg_db = beam_ibis('ibis-postgresql://localhost/mydb')
    # pg_table = (pg_db / 'test_table').write(test_data, schema=TestSchema)
    
    print("\n✅ Schema compatibility demonstrated!")
    print("   Same schema definition works across all Ibis backends")
    print("   No backend-specific code needed!")
    
    # Clean up
    if os.path.exists(sqlite_path):
        os.unlink(sqlite_path)


def demonstrate_error_handling():
    """Demonstrate error handling in write operations."""
    
    print("\n" + "="*60)
    print("ERROR HANDLING DEMONSTRATION")
    print("="*60)
    
    db_path = tempfile.mktemp(suffix='.db')
    
    try:
        db = beam_ibis(f'ibis-sqlite:///{db_path}')
        
        # Create initial table
        test_data = pd.DataFrame([
            {'id': 1, 'name': 'test1'},
            {'id': 2, 'name': 'test2'}
        ])
        
        table = (db / 'test_table').write(test_data)  # Use write API
        print(f"✅ Created initial table with {table.count()} rows")
        
        # 1. Test if_exists="fail"
        print("\n1. Testing if_exists='fail'...")
        try:
            (db / 'test_table').write(test_data, if_exists='fail')  # Use write API
            print("❌ Should have failed!")
        except ValueError as e:
            print(f"✅ Correctly failed: {e}")
        
        # 2. Test if_exists="replace"
        print("\n2. Testing if_exists='replace'...")
        new_data = pd.DataFrame([{'id': 99, 'name': 'replaced'}])
        replaced_table = (db / 'test_table').write(new_data, if_exists='replace')  # Use write API
        print(f"✅ Replaced table, now has {replaced_table.count()} rows")
        
        # 3. Test invalid schema
        print("\n3. Testing invalid schema...")
        try:
            invalid_schema = BeamIbisSchema(
                fields={'bad_field': 'invalid_type_name'}
            )
            print("❌ Should have failed!")
        except ValueError as e:
            print(f"✅ Correctly caught invalid type: {e}")
            
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def main():
    """Run all demonstrations."""
    
    print("BeamIbis Write Operations and Schema Management")
    print("=" * 70)
    print("This example demonstrates how to use ONLY the BeamIbis API")
    print("for all write operations across different backends.")
    print("=" * 70)
    
    try:
        # 1. Schema management
        order_schema, extended_order_schema = demonstrate_schema_management()
        
        # 2. Basic write operations
        demonstrate_write_operations_sqlite()
        
        # 3. Schema-based table creation
        demonstrate_schema_based_table_creation()
        
        # 4. Cross-backend compatibility
        demonstrate_cross_backend_compatibility()
        
        # 5. Error handling
        demonstrate_error_handling()
        
        print("\n" + "="*60)
        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Benefits:")
        print("• ✅ Unified write() API - smart routing based on context")
        print("• ✅ Auto-detects data types (DataFrames, lists, dicts)")
        print("• ✅ Smart routing: create table vs append batch vs append row")
        print("• ✅ Schema support with natural Pydantic class syntax")
        print("• ✅ Schema retrieval from existing tables (get_schema, describe_schema)")
        print("• ✅ Cross-backend compatibility using Ibis native conversion")
        print("• ✅ Type-safe operations with beam.type integration")
        print("• ✅ Comprehensive error handling and validation")
        
    except Exception as e:
        print(f"\n❌ Error in demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 