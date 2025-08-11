"""
Performance tests for data collection pipeline
Target: Execution time benchmarks and memory monitoring
"""

import pytest
import pandas as pd
import time
import psutil
import os
from unittest.mock import Mock, patch


@pytest.mark.performance
class TestCollectionPerformance:
    """Performance benchmarks for data collectors"""
    
    def test_naep_collector_performance_large_request(self):
        """Test NAEP collector performance with large number of requests"""
        from code.collection.naep_collector import NAEPDataCollector
        
        start_memory = psutil.Process(os.getpid()).memory_info().rss
        start_time = time.time()
        
        # Mock large collection scenario
        collector = NAEPDataCollector(rate_limit_delay=0.1)  # Faster for testing
        
        with patch('code.collection.naep_collector.requests.get') as mock_get:
            # Mock fast successful responses
            mock_response = Mock()
            mock_response.json.return_value = {'result': []}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            with patch('code.collection.naep_collector.time.sleep'):
                # Simulate 50 requests (5 years × 2 grades × 5 subjects)
                result = collector.fetch_state_swd_data(
                    years=[2018, 2019, 2020, 2021, 2022],
                    grades=[4, 8],
                    subjects=['mathematics', 'reading', 'science', 'writing', 'civics']
                )
        
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss
        
        execution_time = end_time - start_time
        memory_increase = end_memory - start_memory
        
        # Performance benchmarks
        assert execution_time < 5.0  # Should complete within 5 seconds without actual delays
        assert memory_increase < 50 * 1024 * 1024  # Less than 50MB memory increase
        assert isinstance(result, pd.DataFrame)
        assert mock_get.call_count == 50
        
    def test_validation_performance_large_dataset(self):
        """Test validation performance on large datasets"""
        from code.collection.naep_collector import NAEPDataCollector
        
        # Create large synthetic dataset (10k records)
        large_df = pd.DataFrame({
            'state': (['AL', 'CA', 'TX', 'NY', 'FL'] * 2000),
            'year': ([2019, 2020, 2021, 2022] * 2500),
            'grade': ([4, 8] * 5000),
            'subject': (['mathematics', 'reading'] * 5000),
            'swd_mean': [245.0] * 10000,
            'swd_se': [3.2] * 10000,
            'non_swd_mean': [285.0] * 10000,
            'non_swd_se': [2.1] * 10000,
            'gap': [40.0] * 10000,
            'gap_se': [3.8] * 10000
        })
        
        collector = NAEPDataCollector()
        
        start_time = time.time()
        validation_result = collector.validate_data(large_df)
        end_time = time.time()
        
        validation_time = end_time - start_time
        
        # Validation should be fast even on large datasets
        assert validation_time < 2.0  # Less than 2 seconds for 10k records
        assert validation_result['total_records'] == 10000
        assert isinstance(validation_result, dict)
        
    def test_memory_efficiency_repeated_collections(self):
        """Test memory efficiency across repeated collection cycles"""
        from code.collection.naep_collector import NAEPDataCollector
        
        initial_memory = psutil.Process(os.getpid()).memory_info().rss
        memory_readings = [initial_memory]
        
        collector = NAEPDataCollector()
        
        with patch('code.collection.naep_collector.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                'result': [
                    {
                        'name': 'Alabama',
                        'datavalue': [
                            {'categoryname': 'Students with IEP - Yes', 'value': '245', 'errorFlag': '3.2'},
                            {'categoryname': 'Students with IEP - No', 'value': '285', 'errorFlag': '2.1'}
                        ]
                    }
                ]
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            # Perform 10 collection cycles
            with patch('code.collection.naep_collector.time.sleep'):
                for i in range(10):
                    df = collector.fetch_state_swd_data([2022], [4], ['mathematics'])
                    del df  # Explicit cleanup
                    
                    current_memory = psutil.Process(os.getpid()).memory_info().rss
                    memory_readings.append(current_memory)
        
        final_memory = memory_readings[-1]
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be minimal across repeated operations
        assert memory_growth < 20 * 1024 * 1024  # Less than 20MB growth
        
        # Memory should not continuously increase (no major leaks)
        max_memory = max(memory_readings)
        assert max_memory - initial_memory < 50 * 1024 * 1024  # Peak usage reasonable


@pytest.mark.performance
class TestDataProcessingPerformance:
    """Performance tests for data processing operations"""
    
    def test_large_dataframe_operations(self):
        """Test performance of common DataFrame operations on large datasets"""
        # Create large test DataFrame (100k records)
        large_df = pd.DataFrame({
            'state': (['AL', 'CA', 'TX'] * 33334)[:100000],
            'year': ([2019, 2020, 2021, 2022] * 25000),
            'value': range(100000),
            'category': (['A', 'B'] * 50000)
        })
        
        start_time = time.time()
        
        # Common operations that should be fast
        state_counts = large_df.groupby('state').size()
        year_filter = large_df[large_df['year'] == 2022]
        value_stats = large_df['value'].describe()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Operations should complete quickly
        assert processing_time < 3.0  # Less than 3 seconds for 100k records
        assert len(state_counts) == 3
        assert len(year_filter) > 0
        assert 'mean' in value_stats.index
        
    def test_merge_performance_multiple_sources(self):
        """Test performance of merging data from multiple sources"""
        # Create multiple DataFrames to merge
        base_data = pd.DataFrame({
            'state': ['AL', 'CA', 'TX'] * 10000,
            'year': [2022] * 30000
        })
        
        source1 = base_data.copy()
        source1['naep_score'] = range(30000)
        
        source2 = base_data.copy() 
        source2['inclusion_rate'] = [45.2] * 30000
        
        source3 = base_data.copy()
        source3['spending'] = [12000] * 30000
        
        start_time = time.time()
        
        # Chain merges
        merged = source1.merge(source2, on=['state', 'year'])
        merged = merged.merge(source3, on=['state', 'year'])
        
        end_time = time.time()
        merge_time = end_time - start_time
        
        # Merge should be efficient
        assert merge_time < 2.0  # Less than 2 seconds for 30k record merges
        assert len(merged) == 30000
        assert all(col in merged.columns for col in ['naep_score', 'inclusion_rate', 'spending'])


@pytest.mark.performance
@pytest.mark.slow
class TestEndToEndPerformance:
    """End-to-end performance tests for complete workflows"""
    
    def test_full_pipeline_execution_time(self, temp_data_dir):
        """Test complete pipeline execution time"""
        from code.collection.naep_collector import NAEPDataCollector
        
        start_time = time.time()
        
        # Mock complete pipeline execution
        collectors = {
            'naep': NAEPDataCollector(),
            # Add other collectors when implemented
        }
        
        with patch('code.collection.naep_collector.requests.get') as mock_get:
            # Mock responses for all collectors
            mock_response = Mock()
            mock_response.json.return_value = {'result': []}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            results = {}
            with patch('code.collection.naep_collector.time.sleep'):
                for name, collector in collectors.items():
                    # Simulate collection for multiple years
                    df = collector.fetch_state_swd_data([2019, 2020, 2021, 2022], [4, 8], ['mathematics', 'reading'])
                    validation = collector.validate_data(df)
                    
                    output_path = str(temp_data_dir / "raw" / f"{name}_performance_test.csv")
                    save_success = collector.save_data(df, output_path)
                    
                    results[name] = {
                        'records': len(df),
                        'validation_passed': validation.get('passed', False),
                        'saved': save_success
                    }
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Complete pipeline should execute in reasonable time
        assert total_time < 10.0  # Less than 10 seconds for mock execution
        assert len(results) >= 1
        assert all(result['saved'] for result in results.values())
        
    def test_concurrent_collection_performance(self):
        """Test performance of concurrent data collection"""
        import concurrent.futures
        from code.collection.naep_collector import NAEPDataCollector
        
        def mock_collection_task(year):
            """Mock collection task for concurrent execution"""
            collector = NAEPDataCollector()
            
            with patch('code.collection.naep_collector.requests.get') as mock_get:
                mock_response = Mock()
                mock_response.json.return_value = {'result': []}
                mock_response.raise_for_status.return_value = None
                mock_get.return_value = mock_response
                
                with patch('code.collection.naep_collector.time.sleep'):
                    df = collector.fetch_state_swd_data([year], [4], ['mathematics'])
                    return len(df)
        
        start_time = time.time()
        
        # Test concurrent execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(mock_collection_task, year) for year in [2019, 2020, 2021, 2022]]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        concurrent_time = end_time - start_time
        
        # Concurrent execution should be faster than sequential
        assert concurrent_time < 5.0  # Should complete quickly with mocked requests
        assert len(results) == 4
        assert all(isinstance(result, int) for result in results)


@pytest.mark.performance
class TestResourceMonitoring:
    """Monitor resource usage during data collection"""
    
    def test_cpu_usage_monitoring(self):
        """Monitor CPU usage during intensive operations"""
        import time
        
        # Monitor CPU usage over time
        cpu_readings = []
        start_time = time.time()
        
        # Simulate CPU-intensive operations
        for i in range(5):
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_readings.append(cpu_percent)
            
            # Simulate some work
            df = pd.DataFrame({'data': range(10000)})
            df['squared'] = df['data'] ** 2
            del df
        
        end_time = time.time()
        
        # CPU usage should be reasonable
        avg_cpu = sum(cpu_readings) / len(cpu_readings) if cpu_readings else 0
        max_cpu = max(cpu_readings) if cpu_readings else 0
        
        assert avg_cpu < 90.0  # Average CPU usage under 90%
        assert max_cpu < 95.0  # Peak CPU usage under 95%
        assert end_time - start_time < 2.0  # Quick execution
        
    def test_disk_io_monitoring(self, temp_data_dir):
        """Monitor disk I/O during file operations"""
        import shutil
        
        # Get initial disk usage
        disk_usage_before = shutil.disk_usage(str(temp_data_dir))
        
        # Create and save multiple DataFrames
        for i in range(10):
            df = pd.DataFrame({
                'state': ['AL'] * 1000,
                'year': [2022] * 1000,
                'data': range(1000)
            })
            
            file_path = temp_data_dir / f"test_file_{i}.csv"
            df.to_csv(file_path, index=False)
        
        disk_usage_after = shutil.disk_usage(str(temp_data_dir))
        
        # Verify disk operations completed
        assert disk_usage_before.used <= disk_usage_after.used  # Some disk usage increase expected
        
        # Clean up test files
        for i in range(10):
            file_path = temp_data_dir / f"test_file_{i}.csv"
            if file_path.exists():
                file_path.unlink()