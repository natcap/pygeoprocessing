import contextlib
import logging
import logging.handlers
import os
import queue
import unittest
import unittest.mock
import warnings

from osgeo import gdal
from pygeoprocessing import slurm_utils


def mock_env_var(varname, value):
    try:
        prior_value = os.environ[varname]
    except KeyError:
        prior_value = None

    os.environ[varname] = value
    yield
    os.environ[varname] = prior_value


class SLURMUtilsTest(unittest.TestCase):
    @unittest.mock.patch.dict(os.environ, {"SLURM_MEM_PER_NODE": "128"})
    def test_warning_gdal_cachemax_unset_on_slurm(self):
        """PGP.slurm_utils: test warning when GDAL cache not set on slurm."""
        for gdal_cachesize in [1234567890,  # big number of bytes
                               256]:        # megabytes, exceeds slurm
            with unittest.mock.patch('osgeo.gdal.GetCacheMax',
                                     lambda: gdal_cachesize):
                with unittest.mock.patch('warnings.warn') as warn_mock:
                    slurm_utils.log_warning_if_gdal_will_exhaust_slurm_memory()

                warn_mock.assert_called_once()
                caught_message = warn_mock.call_args[0][0]
                self.assertIn("exceeds the memory SLURM has", caught_message)
                self.assertIn(f"GDAL_CACHEMAX={gdal_cachesize}",
                              caught_message)
                self.assertIn("SLURM_MEM_PER_NODE=128", caught_message)

    @unittest.mock.patch.dict(os.environ, {"SLURM_MEM_PER_NODE": "128"})
    def test_logging_gdal_cachemax_unset_on_slurm(self):
        """PGP.slurm_utils: test logs when GDAL cache not set on slurm."""
        logging_queue = queue.Queue()
        queuehandler = logging.handlers.QueueHandler(logging_queue)
        slurm_logger = logging.getLogger('pygeoprocessing.slurm_utils')
        slurm_logger.addHandler(queuehandler)

        for gdal_cachesize in [1234567890,  # big number of bytes
                               256]:        # megabytes, exceeds slurm
            with unittest.mock.patch('osgeo.gdal.GetCacheMax',
                                     lambda: gdal_cachesize):
                try:
                    logging.captureWarnings(True)  # needed for this test
                    slurm_utils.log_warning_if_gdal_will_exhaust_slurm_memory()
                finally:
                    # Always reset captureWarnings in case of failure so other
                    # tests don't misbehave.
                    logging.captureWarnings(False)

                caught_warnings = []
                while True:
                    try:
                        caught_warnings.append(logging_queue.get_nowait())
                    except queue.Empty:
                        break

                self.assertEqual(len(caught_warnings), 1)
                caught_message = caught_warnings[0].msg
                self.assertIn("exceeds the memory SLURM has", caught_message)
                self.assertIn(f"GDAL_CACHEMAX={gdal_cachesize}",
                              caught_message)
                self.assertIn("SLURM_MEM_PER_NODE=128", caught_message)

        slurm_logger.removeHandler(queuehandler)

    @unittest.mock.patch.dict(os.environ, {}, clear=True)  # clear all env vars
    def test_not_on_slurm_no_warnings(self):
        """PGP.slurm_utils: verify no warnings when not on slurm."""
        with unittest.mock.patch('osgeo.gdal.GetCacheMax',
                                 lambda: 123456789):  # big memory value
            with unittest.mock.patch('warnings.warn') as warn_mock:
                slurm_utils.log_warning_if_gdal_will_exhaust_slurm_memory()

            warn_mock.assert_not_called()

    @unittest.mock.patch.dict(os.environ, {}, clear=True)  # clear all env vars
    def test_not_on_slurm_no_logging(self):
        """PGP.slurm_utils: verify no logging when not on slurm."""
        logging_queue = queue.Queue()
        queuehandler = logging.handlers.QueueHandler(logging_queue)
        slurm_logger = logging.getLogger('pygeoprocessing.slurm_utils')
        slurm_logger.addHandler(queuehandler)

        with unittest.mock.patch('osgeo.gdal.GetCacheMax',
                                 lambda: 123456789):  # big memory value
            try:
                logging.captureWarnings(True)
                slurm_utils.log_warning_if_gdal_will_exhaust_slurm_memory()
            finally:
                logging.captureWarnings(False)
                slurm_logger.removeHandler(queuehandler)

            caught_warnings = []
            while True:
                try:
                    caught_warnings.append(logging_queue.get_nowait())
                except queue.Empty:
                    break

            self.assertEqual(caught_warnings, [])
