#!/usr/bin/env python3
"""
Comprehensive BeamIbis Pathlib API Testing

This script tests all pathlib-like operations within the test_planning dataset:
- Directory operations (iterdir, exists, etc.)
- Query composition and chaining
- GroupBy operations
- Time-based queries
- Composite queries

SAFETY: All operations are restricted to the test_planning dataset only.
"""

import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from beam import resource


def test_pathlib_navigation():
    """Test pathlib-style navigation and directory operations."""
    
    print("\n" + "="*60)
    print("TESTING PATHLIB NAVIGATION")
    print("="*60)
    
    try:
        # Test project root navigation
        print("🏠 Testing project root navigation:")
        root = resource('ibis-bigquery:///algo-agents-ai21')
        print(f"   Root: {root}")
        print(f"   Level: {root.level}")
        print(f"   Path: {root.path}")
        
        # Test dataset navigation
        print("\n📂 Testing dataset navigation:")
        dataset = resource('ibis-bigquery:///algo-agents-ai21/test_planning')
        print(f"   Dataset: {dataset}")
        print(f"   Level: {dataset.level}")
        print(f"   Path: {dataset.path}")
        print(f"   Exists: {dataset.exists()}")
        
        # Test individual table navigation
        print("\n📊 Testing table navigation:")
        table = resource('ibis-bigquery:///algo-agents-ai21/test_planning/sales')
        print(f"   Table: {table}")
        print(f"   Level: {table.level}")
        print(f"   Path: {table.path}")
        print(f"   Exists: {table.exists()}")
        print(f"   Backend: {table.backend}")
        
        return table
        
    except Exception as e:
        print(f"❌ Pathlib navigation test failed: {e}")
        return None


def test_basic_operations(table):
    """Test basic table operations."""
    
    print("\n" + "="*60)
    print("TESTING BASIC OPERATIONS")
    print("="*60)
    
    try:
        print("📏 Basic table information:")
        
        # Test count
        row_count = table.count()
        print(f"   Row count: {row_count:,}")
        
        # Test schema
        schema = table.schema
        print(f"   Columns: {len(schema)}")
        print(f"   Schema: {list(schema.keys())}")
        
        # Test head
        print("\n🔍 Sample data (first 3 rows):")
        sample = table.head(3)
        print(sample)
        
        return True
        
    except Exception as e:
        print(f"❌ Basic operations test failed: {e}")
        return False


def test_filtering_operations(table):
    """Test filtering operations."""
    
    print("\n" + "="*60)
    print("TESTING FILTERING OPERATIONS")
    print("="*60)
    
    try:
        original_count = table.count()
        print(f"📊 Original table: {original_count:,} rows")
        
        # Test equality filter
        print("\n🔍 Equality filtering:")
        electronics = table.with_filter_term('electronics', 'category')
        elec_count = electronics.count()
        print(f"   Electronics items: {elec_count:,} rows")
        
        # Test range filters
        print("\n📊 Range filtering:")
        high_price = table.with_filter_gte(50, 'price')
        high_price_count = high_price.count()
        print(f"   High price items (≥$50): {high_price_count:,} rows")
        
        # Test composite filters
        print("\n🔗 Composite filtering:")
        complex_filter = (table
                         .with_filter_term('electronics', 'category')
                         .with_filter_gte(30, 'price')
                         .with_filter_terms(['US', 'UK'], 'country'))
        
        complex_count = complex_filter.count()
        print(f"   Expensive electronics in US/UK: {complex_count:,} rows")
        
        return True
        
    except Exception as e:
        print(f"❌ Filtering operations test failed: {e}")
        return False


def test_time_based_operations(table):
    """Test time-based filtering and operations."""
    
    print("\n" + "="*60)
    print("TESTING TIME-BASED OPERATIONS")
    print("="*60)
    
    try:
        # Test time range filtering
        print("📅 Time range filtering:")
        week1 = table.with_filter_time_range(
            field='timestamp',
            start='2024-01-01',
            end='2024-01-08'
        )
        week1_count = week1.count()
        print(f"   First week (Jan 1-8): {week1_count:,} rows")
        
        # Test time-based aggregations
        print("\n📊 Time-based aggregations:")
        early = table.with_filter_time_range(
            field='timestamp',
            start='2024-01-01',
            end='2024-01-10'
        )
        early_revenue = early.sum('revenue')
        print(f"   Early period revenue: ${early_revenue:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Time-based operations test failed: {e}")
        return False


def test_aggregation_operations(table):
    """Test aggregation operations."""
    
    print("\n" + "="*60)
    print("TESTING AGGREGATION OPERATIONS")
    print("="*60)
    
    try:
        # Basic aggregations
        print("📊 Basic aggregations:")
        
        print(f"   Count: {table.count():,}")
        print(f"   Sum(revenue): ${table.sum('revenue'):.2f}")
        print(f"   Mean(price): ${table.mean('price'):.2f}")
        print(f"   Min(price): ${table.min('price'):.2f}")
        print(f"   Max(price): ${table.max('price'):.2f}")
        print(f"   Unique users: {table.nunique('user_id'):,}")
        
        # Value counts
        print("\n📈 Value counts:")
        category_counts = table.value_counts('category')
        print("   Category distribution:")
        print(category_counts)
        
        return True
        
    except Exception as e:
        print(f"❌ Aggregation operations test failed: {e}")
        return False


def test_groupby_operations(table):
    """Test GroupBy operations extensively."""
    
    print("\n" + "="*60)
    print("TESTING GROUPBY OPERATIONS")
    print("="*60)
    
    try:
        # Simple groupby
        print("📊 Simple GroupBy:")
        category_stats = table.groupby('category').sum('revenue').mean('price').count()
        category_df = category_stats.as_df()
        print("   Revenue by category:")
        print(category_df)
        
        # Multiple groupby fields
        print("\n🌍 Multi-field GroupBy:")
        country_category = table.groupby(['country', 'category']).sum('revenue').count()
        cc_df = country_category.as_df()
        print("   Revenue by country and category:")
        print(cc_df.head(10))
        
        # Complex aggregations
        print("\n🔢 Complex aggregations:")
        complex_agg = table.groupby('status').agg({
            'revenue': ['sum', 'mean'],
            'quantity': ['sum', 'mean'],
            'price': ['min', 'max'],
            'user_id': 'nunique'
        })
        complex_df = complex_agg.as_df()
        print("   Complex stats by status:")
        print(complex_df)
        
        return True
        
    except Exception as e:
        print(f"❌ GroupBy operations test failed: {e}")
        return False


def test_composite_queries(table):
    """Test complex composite queries."""
    
    print("\n" + "="*60)
    print("TESTING COMPOSITE QUERIES")
    print("="*60)
    
    try:
        # Business intelligence query
        print("📈 Business Intelligence Query:")
        bi_query = (table
                   .with_filter_term('electronics', 'category')
                   .with_filter_gte(40, 'price')
                   .with_filter_terms(['US', 'UK'], 'country')
                   .with_filter_time_range(
                       field='timestamp',
                       start='2024-01-01',
                       end='2024-01-08'
                   )
                   .select('user_id', 'product', 'price', 'country', 'timestamp', 'revenue'))
        
        bi_count = bi_query.count()
        print(f"   Matching records: {bi_count:,}")
        
        if bi_count > 0:
            total_revenue = bi_query.sum('revenue')
            avg_price = bi_query.mean('price')
            unique_users = bi_query.nunique('user_id')
            
            print(f"   Total revenue: ${total_revenue:.2f}")
            print(f"   Average price: ${avg_price:.2f}")
            print(f"   Unique customers: {unique_users}")
        
        # Time series analysis
        print("\n📅 Time Series Analysis:")
        periods = [
            ('2024-01-01', '2024-01-08', 'Week 1'),
            ('2024-01-08', '2024-01-15', 'Week 2'),
            ('2024-01-15', '2024-01-22', 'Week 3'),
        ]
        
        for start, end, label in periods:
            period_data = table.with_filter_time_range(
                field='timestamp',
                start=start,
                end=end
            )
            
            print(f"   {label}: {period_data.count():,} transactions, ${period_data.sum('revenue'):.2f} revenue")
        
        return True
        
    except Exception as e:
        print(f"❌ Composite queries test failed: {e}")
        return False


def test_output_formats(table):
    """Test different output formats."""
    
    print("\n" + "="*60)
    print("TESTING OUTPUT FORMATS")
    print("="*60)
    
    try:
        sample_query = table.with_filter_term('electronics', 'category')
        
        # Test pandas DataFrame
        print("🐼 Pandas DataFrame output:")
        df = sample_query.as_df(limit=3)
        print(f"   Type: {type(df)}")
        print(f"   Shape: {df.shape}")
        print("   Sample:")
        print(df[['product', 'price', 'category']])
        
        # Test dictionary format
        print("\n📄 Dictionary output:")
        dict_data = sample_query.as_dict(limit=2)
        print(f"   Type: {type(dict_data)}")
        print(f"   Count: {len(dict_data) if dict_data else 0}")
        
        return True
        
    except Exception as e:
        print(f"❌ Output formats test failed: {e}")
        return False


def run_comprehensive_test():
    """Run all comprehensive tests."""
    
    print("🧪 COMPREHENSIVE BEAMIBIS PATHLIB API TEST")
    print("="*60)
    print("🎯 Target: test_planning dataset in algo-agents-ai21")
    print("⚠️  SAFETY: All operations restricted to test_planning scope only")
    print("="*60)
    
    # Test navigation first
    table = test_pathlib_navigation()
    
    if table is None:
        print("\n❌ Cannot proceed without table connection")
        return
    
    # Run all tests
    test_functions = [
        test_basic_operations,
        test_filtering_operations,
        test_time_based_operations,
        test_aggregation_operations,
        test_groupby_operations,
        test_composite_queries,
        test_output_formats,
    ]
    
    results = []
    for test_func in test_functions:
        try:
            result = test_func(table)
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"\n❌ {test_func.__name__} crashed: {e}")
            results.append((test_func.__name__, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"📊 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("\n📋 Detailed results:")
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        test_display = test_name.replace('test_', '').replace('_', ' ').title()
        print(f"   {status} {test_display}")
    
    print(f"\n🎯 Test completed! The test_planning dataset is ready for use.")
    print("💡 All tests were safely contained within the test_planning dataset.")


def main():
    """Main function."""
    run_comprehensive_test()


if __name__ == "__main__":
    main() 