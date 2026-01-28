"""
Test Suite for Campaign Manager (P0 Critical)
=============================================
Tests for campaign_manager_v2.py - database persistence layer.

Run with: python -m pytest tests/test_campaign_manager.py -v
Or:       python tests/test_campaign_manager.py
"""

import sys
import os
import tempfile
import shutil
import json
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.campaign_manager_v2 import (
    SCHEMA_VERSION,
    CAMPAIGN_DIR,
    get_schema_version,
    set_schema_version,
    migrate_database,
    check_column_exists,
    get_available_campaigns,
    get_campaign_names,
    check_campaign_exists,
    create_campaign,
    get_campaign_info,
    get_campaign_data,
    save_to_campaign,
    save_cold_flow_result,
    save_hot_fire_result,
    verify_test_data_integrity,
    get_test_traceability,
)


class TestCampaignManager:
    """Test campaign manager database operations."""

    @classmethod
    def setup_class(cls):
        """Create temporary campaign directory for tests."""
        cls.temp_dir = tempfile.mkdtemp(prefix='hda_test_campaigns_')
        cls.original_campaign_dir = CAMPAIGN_DIR

        # Monkey-patch the campaign directory
        import core.campaign_manager_v2 as cm
        cm.CAMPAIGN_DIR = cls.temp_dir

    @classmethod
    def teardown_class(cls):
        """Clean up temporary directory."""
        import core.campaign_manager_v2 as cm
        cm.CAMPAIGN_DIR = cls.original_campaign_dir
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setup_method(self):
        """Clean campaign directory before each test."""
        for f in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, f))

    # =========================================================================
    # Campaign Creation Tests
    # =========================================================================

    def test_create_campaign_cold_flow(self):
        """Test creating a cold flow campaign."""
        db_path = create_campaign(
            campaign_name='test_cf_campaign',
            campaign_type='cold_flow',
            description='Test cold flow campaign'
        )

        assert os.path.exists(db_path)
        assert db_path.endswith('.db')

        # Verify schema
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in c.fetchall()]
        assert 'campaign_info' in tables
        assert 'test_results' in tables

        # Verify campaign info
        c.execute("SELECT campaign_name, campaign_type, schema_version FROM campaign_info")
        info = c.fetchone()
        assert info[0] == 'test_cf_campaign'
        assert info[1] == 'cold_flow'
        assert info[2] == SCHEMA_VERSION

        conn.close()
        print("[PASS] Created cold flow campaign with correct schema")

    def test_create_campaign_hot_fire(self):
        """Test creating a hot fire campaign."""
        db_path = create_campaign(
            campaign_name='test_hf_campaign',
            campaign_type='hot_fire',
            description='Test hot fire campaign'
        )

        assert os.path.exists(db_path)

        # Verify hot fire specific columns exist
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("PRAGMA table_info(test_results)")
        columns = [r[1] for r in c.fetchall()]
        conn.close()

        assert 'avg_pc_bar' in columns
        assert 'avg_thrust_n' in columns
        assert 'avg_isp_s' in columns
        assert 'avg_of_ratio' in columns

        print("[PASS] Created hot fire campaign with correct columns")

    def test_create_duplicate_campaign_fails(self):
        """Test that creating duplicate campaign raises error."""
        create_campaign('duplicate_test', 'cold_flow')

        try:
            create_campaign('duplicate_test', 'cold_flow')
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert 'already exists' in str(e)
            print(f"[PASS] Duplicate campaign rejected: {e}")

    def test_create_campaign_invalid_type_fails(self):
        """Test that invalid campaign type raises error."""
        try:
            create_campaign('invalid_type', 'invalid_type')
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert 'Unknown campaign type' in str(e)
            print(f"[PASS] Invalid campaign type rejected: {e}")

    # =========================================================================
    # Campaign Listing Tests
    # =========================================================================

    def test_get_available_campaigns(self):
        """Test listing available campaigns."""
        create_campaign('campaign_a', 'cold_flow', 'First campaign')
        create_campaign('campaign_b', 'hot_fire', 'Second campaign')

        campaigns = get_available_campaigns()

        assert len(campaigns) == 2
        names = [c['name'] for c in campaigns]
        assert 'campaign_a' in names
        assert 'campaign_b' in names

        cf_campaign = next(c for c in campaigns if c['name'] == 'campaign_a')
        assert cf_campaign['type'] == 'cold_flow'
        assert cf_campaign['test_count'] == 0

        print(f"[PASS] Listed {len(campaigns)} campaigns")

    def test_get_campaign_names(self):
        """Test getting simple campaign name list."""
        create_campaign('name_test_1', 'cold_flow')
        create_campaign('name_test_2', 'cold_flow')

        names = get_campaign_names()

        assert 'name_test_1' in names
        assert 'name_test_2' in names

        print(f"[PASS] Got campaign names: {names}")

    def test_check_campaign_exists(self):
        """Test campaign existence check."""
        create_campaign('exists_test', 'cold_flow')

        assert check_campaign_exists('exists_test') == True
        assert check_campaign_exists('nonexistent') == False

        print("[PASS] Campaign existence check works")

    def test_get_campaign_info(self):
        """Test getting campaign metadata."""
        create_campaign('info_test', 'cold_flow', 'Test description')

        info = get_campaign_info('info_test')

        assert info is not None
        assert info['campaign_name'] == 'info_test'
        assert info['campaign_type'] == 'cold_flow'
        assert info['description'] == 'Test description'
        assert info['schema_version'] == SCHEMA_VERSION

        # Non-existent campaign returns None
        assert get_campaign_info('nonexistent') is None

        print(f"[PASS] Got campaign info: {info['campaign_name']}")

    # =========================================================================
    # Save and Retrieve Tests
    # =========================================================================

    def test_save_to_campaign_basic(self):
        """Test saving basic test data to campaign."""
        create_campaign('save_test', 'cold_flow')

        data = {
            'test_id': 'TEST-001',
            'part': 'INJ-01',
            'serial_num': 'SN-001',
            'avg_p_up_bar': 25.0,
            'avg_mf_g_s': 12.5,
            'avg_cd_CALC': 0.654,
        }

        result = save_to_campaign('save_test', data)
        assert result == True

        # Verify data was saved
        df = get_campaign_data('save_test')
        assert len(df) == 1
        assert df.iloc[0]['test_id'] == 'TEST-001'
        assert df.iloc[0]['avg_cd_CALC'] == 0.654

        print("[PASS] Saved and retrieved basic test data")

    def test_save_to_campaign_with_uncertainties(self):
        """Test saving test data with uncertainties."""
        create_campaign('uncertainty_test', 'cold_flow')

        data = {
            'test_id': 'TEST-002',
            'avg_p_up_bar': 25.0,
            'u_p_up_bar': 0.125,
            'avg_mf_g_s': 12.5,
            'u_mf_g_s': 0.125,
            'avg_cd_CALC': 0.654,
            'u_cd_CALC': 0.018,
            'cd_rel_uncertainty_pct': 2.8,
        }

        save_to_campaign('uncertainty_test', data)

        df = get_campaign_data('uncertainty_test')
        assert df.iloc[0]['u_cd_CALC'] == 0.018
        assert abs(df.iloc[0]['cd_rel_uncertainty_pct'] - 2.8) < 0.01

        print("[PASS] Saved and retrieved uncertainties")

    def test_save_to_campaign_with_traceability(self):
        """Test saving test data with full traceability."""
        create_campaign('trace_test', 'cold_flow')

        data = {
            'test_id': 'TEST-003',
            'avg_cd_CALC': 0.654,
            # Traceability fields
            'raw_data_hash': 'sha256:abc123def456',
            'config_hash': 'sha256:config789',
            'config_snapshot': json.dumps({'name': 'test_config'}),
            'analyst_username': 'test_user',
            'analysis_timestamp_utc': datetime.utcnow().isoformat(),
            'processing_version': '2.0.0',
            # Processing record
            'steady_window_start_ms': 1000.0,
            'steady_window_end_ms': 5000.0,
            'detection_method': 'CV-based',
            # QC
            'qc_passed': 1,
            'qc_summary': json.dumps({'passed': 3, 'failed': 0}),
        }

        save_to_campaign('trace_test', data)

        df = get_campaign_data('trace_test')
        assert df.iloc[0]['raw_data_hash'] == 'sha256:abc123def456'
        assert df.iloc[0]['analyst_username'] == 'test_user'
        assert df.iloc[0]['qc_passed'] == 1

        print("[PASS] Saved and retrieved traceability data")

    def test_save_to_campaign_update_existing(self):
        """Test updating existing test record."""
        create_campaign('update_test', 'cold_flow')

        # Initial save
        data = {'test_id': 'TEST-004', 'avg_cd_CALC': 0.60}
        save_to_campaign('update_test', data)

        # Update
        data['avg_cd_CALC'] = 0.65
        data['comments'] = 'Updated'
        save_to_campaign('update_test', data)

        # Should still have only one record
        df = get_campaign_data('update_test')
        assert len(df) == 1
        assert df.iloc[0]['avg_cd_CALC'] == 0.65
        assert df.iloc[0]['comments'] == 'Updated'

        print("[PASS] Updated existing record correctly")

    def test_save_to_campaign_numpy_types(self):
        """Test saving NumPy types (should be converted)."""
        create_campaign('numpy_test', 'cold_flow')

        data = {
            'test_id': 'TEST-005',
            'avg_cd_CALC': np.float64(0.654),
            'avg_p_up_bar': np.float32(25.0),
            'qc_passed': np.int64(1),
        }

        # Should not raise TypeError
        result = save_to_campaign('numpy_test', data)
        assert result == True

        df = get_campaign_data('numpy_test')
        assert abs(df.iloc[0]['avg_cd_CALC'] - 0.654) < 0.001

        print("[PASS] NumPy types converted and saved correctly")

    def test_save_to_nonexistent_campaign_fails(self):
        """Test saving to non-existent campaign raises error."""
        try:
            save_to_campaign('nonexistent', {'test_id': 'TEST'})
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert 'not found' in str(e)
            print(f"[PASS] Non-existent campaign rejected: {e}")

    # =========================================================================
    # Convenience Save Functions
    # =========================================================================

    def test_save_cold_flow_result(self):
        """Test convenience function for cold flow results."""
        create_campaign('cf_result_test', 'cold_flow')

        result = save_cold_flow_result(
            campaign_name='cf_result_test',
            test_id='CF-001',
            metadata={'part': 'INJ-01', 'serial_num': 'SN-001', 'fluid': 'water'},
            measurements={'avg_p_up_bar': 25.0, 'avg_mf_g_s': 12.5, 'avg_cd_CALC': 0.654},
            uncertainties={'u_p_up_bar': 0.125, 'u_mf_g_s': 0.125, 'u_cd_CALC': 0.018},
            traceability={'raw_data_hash': 'sha256:test', 'config_hash': 'sha256:config'},
            qc_result={'passed': True, 'summary': {'total': 5, 'passed': 5}},
            comments='Test comment'
        )

        assert result == True

        df = get_campaign_data('cf_result_test')
        assert len(df) == 1
        assert df.iloc[0]['part'] == 'INJ-01'
        assert df.iloc[0]['avg_cd_CALC'] == 0.654
        assert df.iloc[0]['qc_passed'] == 1

        print("[PASS] Cold flow convenience save works")

    def test_save_hot_fire_result(self):
        """Test convenience function for hot fire results."""
        create_campaign('hf_result_test', 'hot_fire')

        result = save_hot_fire_result(
            campaign_name='hf_result_test',
            test_id='HF-001',
            metadata={'part': 'ENG-01', 'serial_num': 'SN-001', 'propellants': 'LOX/RP-1'},
            measurements={
                'avg_pc_bar': 50.0,
                'avg_thrust_n': 1000.0,
                'avg_isp_s': 280.0,
                'avg_of_ratio': 2.4,
            },
            uncertainties={
                'u_pc_bar': 0.5,
                'u_thrust_n': 10.0,
                'u_isp_s': 3.0,
                'u_of_ratio': 0.1,
            },
            traceability={'raw_data_hash': 'sha256:hf_test'},
        )

        assert result == True

        df = get_campaign_data('hf_result_test')
        assert df.iloc[0]['avg_isp_s'] == 280.0

        print("[PASS] Hot fire convenience save works")

    # =========================================================================
    # Schema Version and Migration Tests
    # =========================================================================

    def test_get_schema_version(self):
        """Test getting schema version from database."""
        db_path = create_campaign('version_test', 'cold_flow')

        version = get_schema_version(db_path)
        assert version == SCHEMA_VERSION

        print(f"[PASS] Got schema version: {version}")

    def test_set_schema_version(self):
        """Test setting schema version."""
        db_path = create_campaign('set_version_test', 'cold_flow')

        # Set to a different version
        set_schema_version(db_path, 99)

        version = get_schema_version(db_path)
        assert version == 99

        print("[PASS] Set schema version correctly")

    def test_migrate_database_no_migration_needed(self):
        """Test migration when already at latest version."""
        db_path = create_campaign('no_migrate_test', 'cold_flow')

        old_v, new_v = migrate_database(db_path)

        assert old_v == SCHEMA_VERSION
        assert new_v == SCHEMA_VERSION

        print(f"[PASS] No migration needed: v{old_v} -> v{new_v}")

    def test_check_column_exists(self):
        """Test column existence check."""
        db_path = create_campaign('column_test', 'cold_flow')

        assert check_column_exists(db_path, 'test_results', 'test_id') == True
        assert check_column_exists(db_path, 'test_results', 'avg_cd_CALC') == True
        assert check_column_exists(db_path, 'test_results', 'nonexistent_column') == False

        print("[PASS] Column existence check works")

    # =========================================================================
    # Data Retrieval Tests
    # =========================================================================

    def test_get_campaign_data_empty(self):
        """Test getting data from empty campaign."""
        create_campaign('empty_test', 'cold_flow')

        df = get_campaign_data('empty_test')

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

        print("[PASS] Got empty DataFrame from empty campaign")

    def test_get_campaign_data_multiple_tests(self):
        """Test getting data with multiple tests."""
        create_campaign('multi_test', 'cold_flow')

        for i in range(5):
            save_to_campaign('multi_test', {
                'test_id': f'TEST-{i:03d}',
                'avg_cd_CALC': 0.65 + i * 0.01,
                'test_timestamp': datetime.now().isoformat(),
            })

        df = get_campaign_data('multi_test')

        assert len(df) == 5
        assert 'TEST-000' in df['test_id'].values
        assert 'TEST-004' in df['test_id'].values

        print(f"[PASS] Retrieved {len(df)} test records")

    def test_get_campaign_data_nonexistent_fails(self):
        """Test getting data from non-existent campaign."""
        try:
            get_campaign_data('nonexistent')
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert 'not found' in str(e)
            print(f"[PASS] Non-existent campaign query rejected: {e}")

    # =========================================================================
    # Traceability Tests
    # =========================================================================

    def test_get_test_traceability(self):
        """Test retrieving traceability record for a test."""
        create_campaign('trace_retrieve_test', 'cold_flow')

        save_to_campaign('trace_retrieve_test', {
            'test_id': 'TRACE-001',
            'raw_data_path': '/path/to/data.csv',
            'raw_data_hash': 'sha256:abc123',
            'config_hash': 'sha256:config456',
            'analyst_username': 'tester',
            'processing_version': '2.0.0',
            'steady_window_start_ms': 1000.0,
            'steady_window_end_ms': 5000.0,
        })

        trace = get_test_traceability('trace_retrieve_test', 'TRACE-001')

        assert trace is not None
        assert trace['raw_data_hash'] == 'sha256:abc123'
        assert trace['analyst_username'] == 'tester'
        assert trace['steady_window_start_ms'] == 1000.0

        # Non-existent test returns None
        assert get_test_traceability('trace_retrieve_test', 'NONEXISTENT') is None

        print("[PASS] Retrieved traceability record")

    # =========================================================================
    # Edge Cases and Error Handling
    # =========================================================================

    def test_save_with_extra_columns_ignored(self):
        """Test that extra columns not in schema are ignored."""
        create_campaign('extra_col_test', 'cold_flow')

        data = {
            'test_id': 'TEST-EXTRA',
            'avg_cd_CALC': 0.65,
            'nonexistent_column': 'should be ignored',
            'another_fake_column': 12345,
        }

        # Should not raise error
        result = save_to_campaign('extra_col_test', data)
        assert result == True

        df = get_campaign_data('extra_col_test')
        assert len(df) == 1

        print("[PASS] Extra columns ignored gracefully")

    def test_save_with_null_values(self):
        """Test saving with None/null values."""
        create_campaign('null_test', 'cold_flow')

        data = {
            'test_id': 'TEST-NULL',
            'avg_cd_CALC': None,
            'comments': None,
        }

        result = save_to_campaign('null_test', data)
        assert result == True

        df = get_campaign_data('null_test')
        assert pd.isna(df.iloc[0]['avg_cd_CALC'])

        print("[PASS] Null values saved correctly")

    def test_save_with_numpy_bool(self):
        """Test saving np.bool_ values (regression: unsupported type)."""
        create_campaign('npbool_test', 'cold_flow')

        data = {
            'test_id': 'TEST-BOOL',
            'avg_cd_CALC': np.float64(0.654),
            'qc_passed': np.bool_(True),
        }

        result = save_to_campaign('npbool_test', data)
        assert result == True

        df = get_campaign_data('npbool_test')
        assert df.iloc[0]['qc_passed'] == 1

        print("[PASS] np.bool_ type saved correctly")

    def test_save_with_dict_and_list_values(self):
        """Test saving dict/list values (should be JSON-serialized)."""
        create_campaign('dictlist_test', 'cold_flow')

        data = {
            'test_id': 'TEST-DICT',
            'avg_cd_CALC': 0.654,
            'qc_summary': {'total': 5, 'passed': 5, 'failed': 0},
            'detection_parameters': {'threshold': 0.02, 'method': 'cv'},
            'stability_channels': ['PT-01', 'FM-01'],
        }

        result = save_to_campaign('dictlist_test', data)
        assert result == True

        df = get_campaign_data('dictlist_test')
        assert len(df) == 1

        # Dict/list values should have been JSON-serialized
        qc_raw = df.iloc[0]['qc_summary']
        if isinstance(qc_raw, str):
            parsed = json.loads(qc_raw)
            assert parsed['total'] == 5

        print("[PASS] Dict/list values saved correctly as JSON")

    def test_save_full_analysis_result_types(self):
        """Test saving a record with all types from a real analysis result."""
        create_campaign('fulltype_test', 'cold_flow')

        # Simulate what to_database_record + traceability produces
        data = {
            'test_id': 'CF-FULL-001',
            'test_timestamp': datetime.now().isoformat(),
            'qc_passed': np.bool_(True),
            'qc_summary': json.dumps({'total': 5, 'passed': 5}),
            # Metadata
            'part': 'INJ-01',
            'serial_num': 'SN-001',
            'fluid': 'water',
            'operator': 'test_user',
            # Traceability
            'raw_data_hash': 'sha256:abc123',
            'config_hash': 'sha256:config456',
            'config_snapshot': json.dumps({'name': 'test'}),
            'analyst_username': 'tester',
            'processing_version': '2.0.0',
            # Processing record with numpy floats
            'steady_window_start_ms': np.float64(1500.0),
            'steady_window_end_ms': np.float64(5000.0),
            'steady_window_duration_ms': np.float64(3500.0),
            'resample_freq_ms': np.float64(10.0),
            'detection_method': 'CV-based',
            # Measurements with numpy floats
            'avg_p_up_bar': np.float64(25.0),
            'u_p_up_bar': np.float64(0.125),
            'avg_mf_g_s': np.float64(12.5),
            'u_mf_g_s': np.float64(0.125),
            'avg_cd_CALC': np.float64(0.654),
            'u_cd_CALC': np.float64(0.018),
            'cd_rel_uncertainty_pct': np.float64(2.75),
        }

        # This should NOT raise "Error binding parameter" error
        result = save_to_campaign('fulltype_test', data)
        assert result == True

        df = get_campaign_data('fulltype_test')
        assert len(df) == 1
        assert abs(df.iloc[0]['avg_cd_CALC'] - 0.654) < 0.001
        assert df.iloc[0]['qc_passed'] == 1
        assert df.iloc[0]['steady_window_start_ms'] == 1500.0

        print("[PASS] Full analysis result with mixed types saved correctly")

    def test_concurrent_campaign_directory_creation(self):
        """Test that campaign directory is created if missing."""
        import core.campaign_manager_v2 as cm

        # Point to a non-existent directory
        new_temp_dir = os.path.join(self.temp_dir, 'new_subdir')
        original = cm.CAMPAIGN_DIR
        cm.CAMPAIGN_DIR = new_temp_dir

        try:
            # This should create the directory
            campaigns = get_available_campaigns()
            assert os.path.exists(new_temp_dir)
            assert campaigns == []
        finally:
            cm.CAMPAIGN_DIR = original
            shutil.rmtree(new_temp_dir, ignore_errors=True)

        print("[PASS] Campaign directory created automatically")


def run_all_tests():
    """Run all campaign manager tests."""
    print("=" * 60)
    print("Campaign Manager Tests (P0 Critical)")
    print("=" * 60)

    test_class = TestCampaignManager()
    test_class.setup_class()

    methods = [m for m in dir(test_class) if m.startswith('test_')]

    passed = 0
    failed = 0

    for method_name in methods:
        test_class.setup_method()
        method = getattr(test_class, method_name)
        try:
            method()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {method_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    test_class.teardown_class()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
