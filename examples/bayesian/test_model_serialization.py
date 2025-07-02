#!/usr/bin/env python3
"""
Test Model Serialization and Persistence.
Expected to save/load complete optimization state and resume from where it left off.
"""

import numpy as np
import time
import tempfile
import os
from pathlib import Path

from beam.bayesian import BayesianHPOServiceConfig, HPOService
from beam.bayesian.hp_scheme import BaseParameters
from pydantic import Field


class SerializationTestHyperparameters(BaseParameters):
    """Hyperparameters for testing model serialization."""
    learning_rate: float = Field(ge=1e-4, le=1e-1, description="Learning rate")
    batch_size: int = Field(ge=16, le=256, description="Batch size")
    dropout: float = Field(ge=0.0, le=0.5, description="Dropout rate")
    optimizer: str = Field(description="Optimizer", json_schema_extra={
        "enum": ["adam", "sgd", "rmsprop"]
    })


def serialization_objective(h: SerializationTestHyperparameters) -> float:
    """
    Simple objective function for testing serialization.
    Peak around lr=0.01, batch_size=64, dropout=0.2, optimizer=adam.
    """
    lr_effect = -10 * (np.log10(h.learning_rate) + 2)**2  # Optimal around 1e-2
    batch_effect = -(h.batch_size - 64)**2 / 1000         # Optimal at 64
    dropout_effect = -(h.dropout - 0.2)**2 * 10          # Optimal at 0.2
    
    opt_effects = {"adam": 2.0, "rmsprop": 1.0, "sgd": 0.5}
    opt_effect = opt_effects[h.optimizer]
    
    result = lr_effect + batch_effect + dropout_effect + opt_effect
    result += np.random.normal(0, 0.5)  # Add noise
    
    return float(result)


def test_model_serialization():
    """
    Test model serialization and resumption capabilities.
    
    Expected behavior:
    1. Should save complete optimization state (GP model, replay buffer, etc.)
    2. Should load state and resume optimization seamlessly
    3. Performance after loading should match pre-save performance
    4. Should handle different file formats and edge cases
    """
    print("💾 Testing Model Serialization and Persistence")
    print("=" * 70)
    
    # Create temporary directory for saving models
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Phase 1: Initial optimization and save
        print("🔬 Phase 1: Initial Optimization and Save")
        print("-" * 50)
        
        # Initialize service
        hparams = BayesianHPOServiceConfig(
            acquisition_function='LogExpectedImprovement',
            device='cpu',
            buffer_size=1000  # Ensure replay buffer works with serialization
        )
        
        hpo_service = HPOService(hparams=hparams)
        print("✅ HPOService initialized")
        
        # Register problem
        x_scheme = SerializationTestHyperparameters.model_json_schema()
        result = hpo_service.register('serialization_test', x_scheme)
        print(f"✅ Problem registered: {result['message']}")
        
        # Generate initial training data
        print("\n🎲 Generating initial training data...")
        n_init = 20
        x_init = []
        y_init = []
        
        for _ in range(n_init):
            h = SerializationTestHyperparameters(
                learning_rate=10**np.random.uniform(-4, -1),  # 1e-4 to 1e-1
                batch_size=int(np.random.choice([16, 32, 64, 128, 256])),
                dropout=np.random.uniform(0.0, 0.5),
                optimizer=np.random.choice(["adam", "sgd", "rmsprop"])
            )
            y = serialization_objective(h)
            
            x_init.append(dict(h))
            y_init.append(y)
        
        print(f"📊 Generated {n_init} initial samples")
        print(f"📈 Objective range: [{min(y_init):.3f}, {max(y_init):.3f}]")
        
        # Train initial model
        print("\n🔄 Training initial model...")
        result = hpo_service.add('serialization_test', x_init, y_init)
        print(f"✅ {result['message']}")
        
        # Perform some optimization iterations before saving
        print("\n🎯 Performing optimization iterations before save...")
        n_pre_save = 5
        pre_save_results = []
        
        for i in range(n_pre_save):
            result = hpo_service.sample('serialization_test', n_samples=1)
            candidate = result['samples'][0]
            h = SerializationTestHyperparameters(**candidate)
            y = serialization_objective(h)
            
            hpo_service.add('serialization_test', [candidate], [y])
            pre_save_results.append({
                'iteration': i + 1,
                'candidate': candidate,
                'objective': y
            })
            
            print(f"  Iter {i + 1}: y={y:.3f}, lr={candidate['learning_rate']:.2e}, "
                  f"bs={candidate['batch_size']}, dropout={candidate['dropout']:.3f}")
        
        pre_save_best = max(pre_save_results, key=lambda x: x['objective'])
        print(f"\n🏆 Best before save: {pre_save_best['objective']:.3f}")
        
        # Save model state
        print("\n💾 Saving model state...")
        save_path = temp_path / "optimization_state"  # No extension, beam_path will handle it
        
        try:
            # Get the BayesianBeam solver to test save functionality
            solver = hpo_service._problems['serialization_test'].solver
            
            # Use the integrated save_optimization_state method
            solver.save_optimization_state(str(save_path))
            print(f"✅ Model state saved to {save_path}")
            
            # Check what files were created (beam state can create multiple files)
            save_files = list(save_path.parent.glob(f"{save_path.name}*"))
            if save_files:
                total_size = sum(f.stat().st_size for f in save_files)
                print(f"   Created {len(save_files)} state files, total size: {total_size / 1024:.1f} KB")
                for f in save_files:
                    print(f"     {f.name}: {f.stat().st_size / 1024:.1f} KB")
                
                if total_size > 1000:  # At least 1KB total
                    print(f"   ✅ Save files have reasonable size")
                else:
                    print(f"   ⚠️  Save files seem small: {total_size} bytes total")
            else:
                # Check if it's a single file with different extension
                possible_files = list(save_path.parent.glob(f"*{save_path.name}*"))
                if possible_files:
                    print(f"   Found state files: {[f.name for f in possible_files]}")
                else:
                    raise FileNotFoundError("No save files were created")
                
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return
        
        # Phase 2: Load and resume optimization
        print(f"\n🔬 Phase 2: Load Model and Resume Optimization")
        print("-" * 50)
        
        # Create new service instance to simulate fresh start
        hpo_service_new = HPOService(hparams=hparams)
        print("✅ New HPOService instance created")
        
        # Register problem again (in real use, this would be same schema)
        result = hpo_service_new.register('serialization_test_resumed', x_scheme)
        print(f"✅ Problem re-registered: {result['message']}")
        
        # Load model state
        print(f"\n📂 Loading model state from {save_path}...")
        
        try:
            solver_new = hpo_service_new._problems['serialization_test_resumed'].solver
            
            # Use the integrated load_optimization_state method
            solver_new.load_optimization_state(str(save_path))
            print(f"✅ Model state loaded successfully")
            
            # Verify loaded state
            loaded_best = solver_new.best_f
            loaded_samples = len(solver_new.rb)
            
            print(f"📊 Loaded state verification:")
            print(f"   Samples in replay buffer: {loaded_samples}")
            print(f"   Best value: {loaded_best:.3f}")
            print(f"   Has GP model: {solver_new.gp is not None}")
            
            # Compare with pre-save state
            original_samples = len(solver.rb)
            original_best = solver.best_f
            
            print(f"\n🔍 State comparison:")
            print(f"   Original samples: {original_samples} → Loaded: {loaded_samples}")
            print(f"   Original best: {original_best:.3f} → Loaded: {loaded_best:.3f}")
            
            if loaded_samples == original_samples:
                print(f"   ✅ Sample count matches")
            else:
                print(f"   ❌ Sample count mismatch!")
                
            if abs(loaded_best - original_best) < 1e-6:
                print(f"   ✅ Best value matches")
            else:
                print(f"   ❌ Best value mismatch!")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return
        
        # Phase 3: Continue optimization from loaded state
        print(f"\n🎯 Phase 3: Continue Optimization from Loaded State")
        print("-" * 50)
        
        n_post_load = 5
        post_load_results = []
        
        for i in range(n_post_load):
            result = hpo_service_new.sample('serialization_test_resumed', n_samples=1)
            candidate = result['samples'][0]
            h = SerializationTestHyperparameters(**candidate)
            y = serialization_objective(h)
            
            hpo_service_new.add('serialization_test_resumed', [candidate], [y])
            post_load_results.append({
                'iteration': i + 1,
                'candidate': candidate,
                'objective': y
            })
            
            print(f"  Iter {i + 1}: y={y:.3f}, lr={candidate['learning_rate']:.2e}, "
                  f"bs={candidate['batch_size']}, dropout={candidate['dropout']:.3f}")
        
        post_load_best = max(post_load_results, key=lambda x: x['objective'])
        final_best = max(pre_save_best['objective'], post_load_best['objective'])
        
        print(f"\n📊 Optimization continuation results:")
        print(f"   Best before save: {pre_save_best['objective']:.3f}")
        print(f"   Best after resume: {post_load_best['objective']:.3f}")
        print(f"   Overall best: {final_best:.3f}")
        
        # Check if optimization continued effectively
        improvement_after_load = post_load_best['objective'] - loaded_best
        print(f"   Improvement after load: {improvement_after_load:.3f}")
        
        if improvement_after_load > 0:
            print(f"   ✅ Optimization improved after loading")
        else:
            print(f"   ⚠️  No improvement after loading (this can be normal)")
        
        # Test multiple save/load cycles
        print(f"\n🔄 Testing multiple save/load cycles...")
        save_path_2 = temp_path / "optimization_state_2"
        
        try:
            # Save current state using integrated method
            solver_new.save_optimization_state(str(save_path_2))
            print(f"   ✅ Second save completed")
            
            # Check file sizes for both saves
            save_files_1 = list(save_path.parent.glob(f"{save_path.name}*"))
            save_files_2 = list(save_path_2.parent.glob(f"{save_path_2.name}*"))
            
            if save_files_1 and save_files_2:
                size1 = sum(f.stat().st_size for f in save_files_1)
                size2 = sum(f.stat().st_size for f in save_files_2)
                print(f"   Save file sizes: {size1/1024:.1f} KB → {size2/1024:.1f} KB")
                
                if size2 >= size1:
                    print(f"   ✅ Second save size is appropriate (accumulated more data)")
                else:
                    print(f"   ⚠️  Second save is smaller than first")
                
        except Exception as e:
            print(f"   ❌ Error in second save: {e}")
    
    print(f"\n✅ Model serialization test completed!")
    
    # Expected Outputs
    print(f"\n📋 Expected Test Outcomes:")
    print(f"   • Should save and load complete optimization state without errors")
    print(f"   • Loaded state should match original state (samples, best value)")
    print(f"   • Optimization should continue seamlessly after loading")
    print(f"   • Save files should have reasonable sizes (>1KB)")
    print(f"   • Multiple save/load cycles should work")
    print(f"   • GP model state should be preserved and functional")
    print(f"   • Replay buffer should be fully restored")


if __name__ == "__main__":
    test_model_serialization() 