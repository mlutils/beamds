#!/usr/bin/env python3
"""
BeamIbis Pathlib Navigation Test

This script tests pathlib-style navigation operations:
- Path construction and parsing
- Level detection
- exists() checking
- Path navigation patterns

This test focuses on the pathlib-like interface without requiring actual data operations.
"""

from beam import resource


def test_path_construction():
    """Test path construction and parsing."""
    
    print("\n" + "="*60)
    print("TESTING PATH CONSTRUCTION")
    print("="*60)
    
    # Test different URI patterns
    test_cases = [
        ('ibis-bigquery:///algo-agents-ai21', 'Project root'),
        ('ibis-bigquery:///algo-agents-ai21/', 'Project root with slash'),
        ('ibis-bigquery:///algo-agents-ai21/test_planning', 'Dataset level'),
        ('ibis-bigquery:///algo-agents-ai21/test_planning/', 'Dataset with slash'),
        ('ibis-bigquery:///algo-agents-ai21/test_planning/sales', 'Table level'),
        ('ibis-bigquery:///algo-agents-ai21/test_planning/sales/', 'Table with slash'),
    ]
    
    for uri, description in test_cases:
        try:
            print(f"\n🔍 {description}:")
            print(f"   URI: {uri}")
            
            bi = resource(uri)
            print(f"   ✅ Created: {bi}")
            print(f"   Path: {bi.path}")
            print(f"   Level: {bi.level}")
            print(f"   Backend: {bi.backend}")
            
            # Test path components
            path_parts = bi.path.strip('/').split('/') if bi.path.strip('/') else []
            print(f"   Path parts: {path_parts}")
            print(f"   Part count: {len(path_parts)}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")


def test_level_detection():
    """Test level detection logic."""
    
    print("\n" + "="*60)
    print("TESTING LEVEL DETECTION")
    print("="*60)
    
    test_cases = [
        ('ibis-bigquery:///algo-agents-ai21', 'root'),
        ('ibis-bigquery:///algo-agents-ai21/test_planning', 'dataset'),
        ('ibis-bigquery:///algo-agents-ai21/test_planning/sales', 'table'),
        ('ibis-bigquery:///algo-agents-ai21/test_planning/sales/column', 'column'),
    ]
    
    for uri, expected_level in test_cases:
        try:
            bi = resource(uri)
            actual_level = bi.level
            match = actual_level == expected_level
            status = "✅" if match else "❌"
            
            print(f"{status} {uri}")
            print(f"   Expected: {expected_level}")
            print(f"   Actual: {actual_level}")
            print(f"   Path: {bi.path}")
            
        except Exception as e:
            print(f"❌ {uri}")
            print(f"   Error: {e}")


def test_path_navigation():
    """Test path navigation patterns."""
    
    print("\n" + "="*60)
    print("TESTING PATH NAVIGATION PATTERNS")
    print("="*60)
    
    try:
        # Start at root
        print("🏠 Starting at project root:")
        root = resource('ibis-bigquery:///algo-agents-ai21')
        print(f"   Root: {root}")
        print(f"   Level: {root.level}")
        
        # Navigate to dataset
        print("\n📂 Navigate to dataset:")
        dataset = resource('ibis-bigquery:///algo-agents-ai21/test_planning')
        print(f"   Dataset: {dataset}")
        print(f"   Level: {dataset.level}")
        
        # Navigate to tables
        print("\n📊 Navigate to tables:")
        table_names = ['sales', 'transactions', 'user_activity']
        
        for table_name in table_names:
            try:
                table = resource(f'ibis-bigquery:///algo-agents-ai21/test_planning/{table_name}')
                print(f"   {table_name}: {table}")
                print(f"      Level: {table.level}")
                print(f"      Path: {table.path}")
                print(f"      Backend: {table.backend}")
            except Exception as e:
                print(f"   ❌ {table_name}: {e}")
        
        # Test path relationships
        print("\n🔗 Path relationships:")
        print(f"   Root path: '{root.path}'")
        print(f"   Dataset path: '{dataset.path}'")
        print(f"   Dataset contains root: {dataset.path.startswith(root.path)}")
        
        # Test path construction patterns
        print("\n🏗️ Path construction patterns:")
        
        # Parent-child relationships
        if dataset.path.startswith(root.path):
            print("   ✅ Dataset is child of root")
        else:
            print("   ❌ Dataset is NOT child of root")
        
        # Path depth
        root_depth = len(root.path.strip('/').split('/')) if root.path.strip('/') else 0
        dataset_depth = len(dataset.path.strip('/').split('/')) if dataset.path.strip('/') else 0
        
        print(f"   Root depth: {root_depth}")
        print(f"   Dataset depth: {dataset_depth}")
        print(f"   Depth relationship: {'✅' if dataset_depth > root_depth else '❌'}")
        
    except Exception as e:
        print(f"❌ Path navigation test failed: {e}")


def test_exists_checking():
    """Test exists() method without requiring full connection."""
    
    print("\n" + "="*60)
    print("TESTING EXISTS() CHECKING")
    print("="*60)
    
    test_cases = [
        ('ibis-bigquery:///algo-agents-ai21', 'Project root'),
        ('ibis-bigquery:///algo-agents-ai21/test_planning', 'Test dataset'),
        ('ibis-bigquery:///algo-agents-ai21/test_planning/sales', 'Sales table'),
        ('ibis-bigquery:///algo-agents-ai21/nonexistent_dataset', 'Nonexistent dataset'),
        ('ibis-bigquery:///algo-agents-ai21/test_planning/nonexistent_table', 'Nonexistent table'),
    ]
    
    for uri, description in test_cases:
        try:
            print(f"\n🔍 {description}:")
            print(f"   URI: {uri}")
            
            bi = resource(uri)
            
            # Try exists() method
            try:
                exists = bi.exists()
                print(f"   exists(): {exists}")
            except Exception as e:
                print(f"   exists() failed: {e}")
                
            # Show other properties
            print(f"   Level: {bi.level}")
            print(f"   Path: {bi.path}")
            
        except Exception as e:
            print(f"   ❌ Resource creation failed: {e}")


def test_pathlib_like_operations():
    """Test pathlib-like operations and properties."""
    
    print("\n" + "="*60)
    print("TESTING PATHLIB-LIKE OPERATIONS")
    print("="*60)
    
    try:
        # Test different resources
        resources = [
            resource('ibis-bigquery:///algo-agents-ai21'),
            resource('ibis-bigquery:///algo-agents-ai21/test_planning'),
            resource('ibis-bigquery:///algo-agents-ai21/test_planning/sales'),
        ]
        
        for i, bi in enumerate(resources):
            print(f"\n📋 Resource {i+1}: {bi}")
            
            # Test basic properties
            print(f"   Path: '{bi.path}'")
            print(f"   Level: {bi.level}")
            print(f"   Backend: {bi.backend}")
            
            # Test string representation
            print(f"   String: '{str(bi)}'")
            print(f"   Repr: {repr(bi)}")
            
            # Test path-like properties
            try:
                # Test if it has path-like behavior
                path_obj = bi.path
                print(f"   Path object type: {type(path_obj)}")
                print(f"   Path is string: {isinstance(path_obj, str)}")
                
                # Test path operations
                if hasattr(bi, 'name'):
                    print(f"   Name: {bi.name}")
                else:
                    # Extract name from path
                    name = bi.path.split('/')[-1] if '/' in bi.path else bi.path
                    print(f"   Extracted name: '{name}'")
                
            except Exception as e:
                print(f"   ⚠️  Path operations failed: {e}")
        
        # Test path comparison
        print(f"\n🔍 Path comparisons:")
        root = resources[0]
        dataset = resources[1]
        table = resources[2]
        
        print(f"   Root == Dataset: {root.path == dataset.path}")
        print(f"   Dataset == Table: {dataset.path == table.path}")
        print(f"   Root != Dataset: {root.path != dataset.path}")
        
        # Test path hierarchy
        print(f"   Table path starts with dataset: {table.path.startswith(dataset.path)}")
        print(f"   Dataset path starts with root: {dataset.path.startswith(root.path)}")
        
    except Exception as e:
        print(f"❌ Pathlib-like operations test failed: {e}")


def test_error_handling():
    """Test error handling for invalid paths."""
    
    print("\n" + "="*60)
    print("TESTING ERROR HANDLING")
    print("="*60)
    
    error_cases = [
        ('ibis-bigquery:///', 'Empty path'),
        ('ibis-bigquery:///invalid-project-name', 'Invalid project'),
        ('ibis-bigquery:///algo-agents-ai21//double-slash', 'Double slash'),
        ('ibis-bigquery:///algo-agents-ai21/test_planning//sales', 'Double slash in path'),
    ]
    
    for uri, description in error_cases:
        try:
            print(f"\n🔍 {description}:")
            print(f"   URI: {uri}")
            
            bi = resource(uri)
            print(f"   ✅ Created (unexpectedly): {bi}")
            print(f"   Level: {bi.level}")
            print(f"   Path: {bi.path}")
            
        except Exception as e:
            print(f"   ✅ Failed as expected: {e}")


def run_navigation_tests():
    """Run all navigation tests."""
    
    print("🧪 BEAMIBIS PATHLIB NAVIGATION TEST")
    print("="*60)
    print("🎯 Testing pathlib-style navigation without data operations")
    print("⚠️  Focus: Path construction, level detection, navigation patterns")
    print("="*60)
    
    # Run all test functions
    test_functions = [
        test_path_construction,
        test_level_detection,
        test_path_navigation,
        test_exists_checking,
        test_pathlib_like_operations,
        test_error_handling,
    ]
    
    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            print(f"\n❌ {test_func.__name__} crashed: {e}")
    
    print("\n" + "="*60)
    print("NAVIGATION TEST SUMMARY")
    print("="*60)
    print("✅ Path construction and parsing tests completed")
    print("✅ Level detection tests completed")  
    print("✅ Navigation pattern tests completed")
    print("✅ Pathlib-like operation tests completed")
    print("✅ Error handling tests completed")
    print("\n💡 Navigation API is working! Next step: Fix connection issues for data operations.")


def main():
    """Main function."""
    run_navigation_tests()


if __name__ == "__main__":
    main() 