import unittest
from unittest.mock import patch
from training import lmb4_qualified_source as S

class QualifiedSourceGuardTests(unittest.TestCase):
    def invoke(self,receipt,report):
        with patch.object(S.R,'verify',return_value={}),patch.object(S.R.L.P.L3,'read',side_effect=[receipt,report]),patch.object(S.R.L.P.L3,'sha',return_value='actual'):
            return S.qualified_source()

    def test_raw_pass_cannot_replace_missing_independent_receipt(self):
        with patch.object(S.R,'verify',return_value={}),patch.object(S.R.L.P.L3,'read',side_effect=FileNotFoundError('audit absent')):
            with self.assertRaises(FileNotFoundError):S.qualified_source()

    def test_failed_or_unqualified_audit_cannot_return_parent(self):
        for receipt in (dict(status='FAIL',verdict='PASS',qualification=True),dict(status='PASS',verdict='FAIL',qualification=False),dict(status='PASS',verdict='PASS',qualification=False)):
            with self.assertRaises(AssertionError):self.invoke(receipt,dict(verdict='PASS'))

    def test_changed_raw_report_breaks_audited_source_identity(self):
        with self.assertRaises(AssertionError):self.invoke(dict(status='PASS',verdict='PASS',qualification=True,report_sha='old'),dict(verdict='PASS'))

if __name__=='__main__':unittest.main()
