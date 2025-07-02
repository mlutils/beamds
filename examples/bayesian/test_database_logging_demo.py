"""
Demo test for database logging functionality in Bayesian optimization.

This test demonstrates:
1. Setting up database logging with SQLite
2. Running optimization with automatic logging
3. Querying experiment statistics from the database
4. Manual result logging with timing information
"""

import os
import tempfile
import time
from pathlib import Path

def test_database_logging_demo():
    """Test database logging functionality with a complete workflow."""
    
    print("=" * 60)
    print("BAYESIAN OPTIMIZATION DATABASE LOGGING DEMO")
    print("=" * 60)
    
    # Create a temporary SQLite database for this demo
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "optimization_experiments.db"
        dataset_uri = f"ibis-sqlite:///{db_path}/experiment_logs"
        
        print(f"Database URI: {dataset_uri}")
        print(f"Database file: {db_path}")
        
        # 1. Setup HPOService with database logging
        print("\n1. Setting up HPOService with database logging...")
        
        from beam.bayesian.hpo_service import HPOService
        from beam.bayesian.config import BayesianHPOServiceConfig
        
        config = BayesianHPOServiceConfig(
            dataset=dataset_uri,
            experiment_name="database_logging_demo",
            log_suggestions=True,
            log_results=True,
            start_fitting_after_n_points=3,  # Small for demo
            fit_every_n_points=10,
            device="cpu"
        )
        
        service = HPOService(hparams=config)
        print(f"✅ HPOService created with database logging enabled")
        
        # 2. Register optimization problem
        print("\n2. Registering optimization problem...")
        
        # Simple 2D optimization problem
        x_scheme = {
            "type": "object",
            "properties": {
                "x": {"type": "number", "minimum": -5.0, "maximum": 5.0},
                "y": {"type": "number", "minimum": -5.0, "maximum": 5.0}
            },
            "required": ["x", "y"]
        }
        
        # Multi-objective with constraints
        y_scheme = {
            "type": "object", 
            "properties": {
                "objective1": {"type": "number", "objective": "maximize"},
                "objective2": {"type": "number", "objective": "minimize"},
                "constraint1": {"type": "number", "constraint": "<= 10.0"}
            },
            "required": ["objective1", "objective2", "constraint1"]
        }
        
        result = service.register(
            name="demo_problem",
            x_scheme=x_scheme,
            y_scheme=y_scheme
        )
        print(f"✅ Problem registered: {result['message']}")
        print(f"   Objectives: {result['objectives']}")
        print(f"   Constraints: {result['constraints']}")
        
        # 3. Generate initial samples (will be logged automatically)
        print("\n3. Generating initial samples...")
        
        response = service.sample("demo_problem", n_samples=1)
        print(f"✅ Initial samples generated: {response['method']}")
        print(f"   Logged suggestions: {response.get('logged_suggestions', 0)}")
        print(f"   Sample: {response['samples'][0] if response['samples'] else 'None'}")
        
        # 4. Evaluate and add results (simulated evaluation)
        print("\n4. Evaluating samples and adding results...")
        
        def evaluate_sample(x_val, y_val):
            """Simulate a multi-objective function evaluation."""
            time.sleep(0.1)  # Simulate computation time
            
            # Objective 1: maximize -(x^2 + y^2) (inverted sphere function)
            obj1 = -(x_val**2 + y_val**2)
            
            # Objective 2: minimize |x + y| (to conflict with obj1)
            obj2 = abs(x_val + y_val)
            
            # Constraint: sum of absolute values <= 10
            constraint1 = abs(x_val) + abs(y_val)
            
            return {
                "objective1": obj1,
                "objective2": obj2, 
                "constraint1": constraint1
            }
        
        # Evaluate initial samples
        x_data = []
        y_data = []
        
        for sample in response['samples']:
            start_time = time.time()
            result = evaluate_sample(sample['x'], sample['y'])
            execution_time = time.time() - start_time
            
            x_data.append(sample)
            y_data.append(result)
            
            print(f"   Sample {sample} -> {result} (took {execution_time:.3f}s)")
        
        # Add results to service (will be logged automatically)
        add_result = service.add("demo_problem", x_data, y_data)
        print(f"✅ Results added: {add_result['message']}")
        print(f"   Logged results: {add_result.get('logged_results', 0)}")
        
        # 5. Generate a few more samples to show optimization
        print("\n5. Running optimization iterations...")
        
        for iteration in range(3):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Get optimized suggestions
            response = service.sample("demo_problem", n_samples=1)
            print(f"   Method: {response['method']}")
            print(f"   Logged suggestions: {response.get('logged_suggestions', 0)}")
            
            if response['samples']:
                sample = response['samples'][0]
                print(f"   Suggested sample: {sample}")
                
                # Evaluate
                start_time = time.time()
                result = evaluate_sample(sample['x'], sample['y'])
                execution_time = time.time() - start_time
                
                print(f"   Evaluation result: {result}")
                print(f"   Execution time: {execution_time:.3f}s")
                
                # Add result
                add_result = service.add("demo_problem", [sample], [result])
                
                # Also demonstrate manual logging with timing
                manual_log = service.log_result(
                    name="demo_problem",
                    parameters=sample,
                    objectives=result,
                    execution_time=execution_time,
                    success=True
                )
                print(f"   Manual log result: {manual_log['message']}")
        
        # 6. Query experiment statistics
        print("\n6. Querying experiment statistics...")
        
        stats = service.get_experiment_summary("demo_problem")
        print(f"✅ Experiment statistics:")
        print(f"   Total suggestions: {stats.get('total_suggestions', 0)}")
        print(f"   Total results: {stats.get('total_results', 0)}")
        print(f"   Success rate: {stats.get('success_rate', 0):.2%}")
        print(f"   Average execution time: {stats.get('avg_execution_time', 0):.3f}s")
        print(f"   Total execution time: {stats.get('total_execution_time', 0):.3f}s")
        
        # 7. Demonstrate direct database access
        print("\n7. Direct database access...")
        
        try:
            from beam import resource
            
            # Connect to the database
            db = resource(dataset_uri)
            
            # List available tables
            dataset_level = db.parent if db.level == 'table' else db
            tables = list(dataset_level.iterdir())
            print(f"   Available tables: {[t.name for t in tables]}")
            
            # Query suggestions table
            suggestions_table = dataset_level / "suggestions"
            if suggestions_table.exists():
                suggestions_df = suggestions_table.as_df()
                print(f"   Suggestions table: {len(suggestions_df)} rows")
                print(f"   Columns: {list(suggestions_df.columns)}")
                
                if len(suggestions_df) > 0:
                    print("   Sample suggestion record:")
                    first_row = suggestions_df.iloc[0]
                    print(f"     Problem: {first_row.get('problem_name', 'N/A')}")
                    print(f"     Iteration: {first_row.get('iteration', 'N/A')}")
                    print(f"     Acquisition function: {first_row.get('acquisition_function', 'N/A')}")
                    print(f"     Is initial: {first_row.get('is_initial', 'N/A')}")
            
            # Query results table
            results_table = dataset_level / "results"
            if results_table.exists():
                results_df = results_table.as_df()
                print(f"   Results table: {len(results_df)} rows")
                
                if len(results_df) > 0:
                    print("   Sample result record:")
                    first_row = results_df.iloc[0]
                    print(f"     Problem: {first_row.get('problem_name', 'N/A')}")
                    print(f"     Success: {first_row.get('success', 'N/A')}")
                    print(f"     Multi-objective: {first_row.get('is_multi_objective', 'N/A')}")
                    print(f"     Execution time: {first_row.get('execution_time', 'N/A'):.3f}s")
        
        except Exception as e:
            print(f"   ❌ Direct database access failed: {e}")
        
        print("\n" + "=" * 60)
        print("DATABASE LOGGING DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Database file created at: {db_path}")
        print("This demonstrates:")
        print("• Automatic logging of suggestions and results")
        print("• Multi-objective optimization with constraints")
        print("• Manual result logging with timing")
        print("• Experiment statistics querying")
        print("• Direct database access via BeamIbis")


if __name__ == "__main__":
    test_database_logging_demo() 