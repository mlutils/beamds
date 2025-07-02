#!/usr/bin/env python3
"""
BeamIbis Schema Usage Example

This example shows:
1. When schemas are useful vs when they're not needed
2. How to use the new natural Pydantic class syntax for schemas
3. The benefits of using schemas vs creating tables from data
"""

import pandas as pd
import tempfile
import os
from datetime import datetime

# Import BeamIbis components
from beam.sql import beam_ibis, BeamIbisSchema, OrderSchema


def demonstrate_no_schema_needed():
    """Show that schemas aren't always needed - you can create tables from data."""
    
    print("\n" + "="*60)
    print("CASE 1: NO SCHEMA NEEDED - CREATE FROM DATA")
    print("="*60)
    
    # Create temporary database
    db_path = tempfile.mktemp(suffix='.db')
    
    try:
        db = beam_ibis(f'ibis-sqlite:///{db_path}')
        
        # Create sample data - schemas are inferred automatically
        sample_data = pd.DataFrame([
            {'id': 1, 'name': 'Alice', 'age': 25, 'created': datetime.now()},
            {'id': 2, 'name': 'Bob', 'age': 30, 'created': datetime.now()},
            {'id': 3, 'name': 'Charlie', 'age': 35, 'created': datetime.now()},
        ])
        
        print("✅ Creating table directly from DataFrame...")
        print(f"   Data shape: {sample_data.shape}")
        print(f"   Data types: {sample_data.dtypes.to_dict()}")
        
        # Create table from data - no schema needed!
        users_table = (db / 'users').write(sample_data)  # Use write API
        
        print(f"✅ Table created with {users_table.count()} rows")
        print(f"   Inferred schema: {users_table.schema}")
        
        # Query the data
        result = users_table.with_filter_gt(28, 'age').as_df()
        print(f"✅ Users over 28: {len(result)} found")
        print(result)
        
        print("\n💡 When you DON'T need schemas:")
        print("   • You have existing data (DataFrame, CSV, etc.)")
        print("   • Data types can be inferred automatically") 
        print("   • You're doing exploratory data analysis")
        print("   • One-off data imports")
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def demonstrate_schema_benefits():
    """Show when schemas are useful and how to use the new syntax."""
    
    print("\n" + "="*60) 
    print("CASE 2: SCHEMAS ARE USEFUL - NATURAL PYDANTIC SYNTAX")
    print("="*60)
    
    # Define schema using natural Pydantic class syntax
    class ProductSchema(BeamIbisSchema):
        """Product catalog schema with proper types and constraints."""
        id: int
        name: str
        price: float
        category: str
        created_at: datetime
        is_available: bool = True  # Default value
    
    print("✅ Schema defined using natural Python class syntax:")
    print(f"   Class: {ProductSchema.__name__}")
    print(f"   Fields: {ProductSchema.get_field_names()}")
    print(f"   Ibis schema: {ProductSchema.to_ibis_schema()}")
    
    # Create temporary database
    db_path = tempfile.mktemp(suffix='.db')
    
    try:
        db = beam_ibis(f'ibis-sqlite:///{db_path}')
        
        # 1. Create empty table from schema using write API
        print("\n1. Creating empty table from schema...")
        products_table = (db / 'products').write([], schema=ProductSchema)  # Use write API with schema
        print(f"✅ Empty table created: {products_table}")
        print(f"   Schema enforced: {products_table.schema}")
        
        # 2. Generate sample data that matches schema
        print("\n2. Generating sample data that matches schema...")
        sample_data = ProductSchema.create_sample_data(5)
        
        # Convert to DataFrame with proper types
        df = pd.DataFrame(sample_data)
        print(f"✅ Generated {len(df)} sample records")
        print(df.head(3))
        
        # 3. Insert data into the schema-defined table using write API
        print("\n3. Inserting data into schema-defined table...")
        products_table.write(df)  # Smart routing to append_batch
        print(f"✅ Data inserted. Total rows: {products_table.count()}")
        
        # 4. Use predefined schema for consistency
        print("\n4. Using predefined OrderSchema...")
        print(f"   OrderSchema fields: {OrderSchema.get_field_names()}")
        
        orders_table = (db / 'orders').write([], schema=OrderSchema)  # Create from schema
        
        # Add a sample order using write API
        sample_order = {
            'order_id': 'ORD_001',
            'customer_id': 'CUST_001', 
            'product_id': 'PROD_001',
            'quantity': 2,
            'unit_price': 29.99,
            'total_amount': 59.98,
            'order_date': datetime.now(),
            'is_paid': True
        }
        
        orders_table.write(sample_order)  # Smart routing to append_row
        print(f"✅ Order added. Total orders: {orders_table.count()}")
        
        # 5. Demonstrate schema retrieval
        print("\n5. Schema retrieval capabilities...")
        print("✅ Products table schema:")
        print(products_table.describe_schema())
        
        print("\n✅ Orders table schema:")
        print(orders_table.describe_schema())
        
        print("\n💡 When schemas ARE useful:")
        print("   • Creating empty tables with specific structure")
        print("   • Ensuring data type consistency across environments")
        print("   • API contracts and data validation")
        print("   • Reusable table definitions")
        print("   • Cross-backend compatibility")
        print("   • Team collaboration with clear data contracts")
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def demonstrate_schema_evolution():
    """Show how to evolve schemas using inheritance."""
    
    print("\n" + "="*60)
    print("CASE 3: SCHEMA EVOLUTION WITH INHERITANCE")
    print("="*60)
    
    # Base schema
    class BaseUserSchema(BeamIbisSchema):
        """Basic user schema."""
        id: int
        email: str
        created_at: datetime
    
    # Extended schema with more fields
    class ExtendedUserSchema(BaseUserSchema):
        """Extended user schema with additional fields."""
        name: str
        age: int
        is_premium: bool = False
        preferences_json: str = "{}"  # Store as JSON string for SQLite compatibility
    
    print("✅ Schema evolution using inheritance:")
    print(f"   Base schema: {BaseUserSchema.get_field_names()}")
    print(f"   Extended schema: {ExtendedUserSchema.get_field_names()}")
    
    # Create database
    db_path = tempfile.mktemp(suffix='.db')
    
    try:
        db = beam_ibis(f'ibis-sqlite:///{db_path}')
        
        # Create table with extended schema using write API
        users_table = (db / 'users').write([], schema=ExtendedUserSchema)  # Create from schema
        
        # Add sample data using write API
        sample_users = ExtendedUserSchema.create_sample_data(3)
        df = pd.DataFrame(sample_users)
        users_table.write(df)  # Smart routing to append_batch
        
        print(f"✅ Created table with extended schema: {users_table.count()} users")
        
        # Query the data and show schema
        premium_users = users_table.with_filter_term(True, 'is_premium').as_df()
        print(f"✅ Premium users: {len(premium_users)}")
        
        # Demonstrate schema retrieval
        print("\n✅ Extended schema details:")
        print(users_table.describe_schema())
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def main():
    """Run all demonstrations."""
    
    print("BeamIbis Schema Usage Guide")
    print("=" * 60)
    print("When to use schemas vs when you don't need them")
    print("=" * 60)
    
    try:
        # Show when schemas aren't needed
        demonstrate_no_schema_needed()
        
        # Show when schemas are beneficial
        demonstrate_schema_benefits()
        
        # Show schema evolution
        demonstrate_schema_evolution()
        
        print("\n" + "="*60)
        print("✅ SUMMARY")
        print("="*60)
        print()
        print("🚫 You DON'T need schemas when:")
        print("   • Creating tables from existing data")
        print("   • Doing exploratory data analysis")
        print("   • One-off data imports")
        print()
        print("✅ Schemas ARE useful when:")
        print("   • Creating empty tables with specific structure")
        print("   • Ensuring type consistency across environments")
        print("   • Building reusable data contracts")
        print("   • Team collaboration")
        print("   • Cross-backend compatibility")
        print()
        print("🎯 New Unified Write API:")
        print("   • Smart routing: table.write(data) detects context automatically")
        print("   • Supports DataFrames, lists of dicts, single dicts")
        print("   • Schema parameter: table.write(data, schema=MySchema)")
        print("   • Schema retrieval: table.get_schema(), table.describe_schema()")
        print("   • Type detection using beam.type.check_type")
        print("   • Cross-backend compatibility with same API")
        
    except Exception as e:
        print(f"\n❌ Error in demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 