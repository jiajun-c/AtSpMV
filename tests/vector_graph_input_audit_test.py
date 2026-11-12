import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

path=Path(__file__).resolve().parents[1]/'scripts/analysis/audit_vector_graph_inputs.py'
spec=importlib.util.spec_from_file_location('graph_input_audit',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class GraphInputAuditTest(unittest.TestCase):
    def test_changed_inputs_and_pending_are_distinguished(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);ref=base/'ref';run=base/'run'
            (ref/'status').mkdir(parents=True);(run/'status').mkdir(parents=True)
            (run/'manifest.json').write_text(json.dumps([{'dataset':'a'},{'dataset':'b'}]))
            (ref/'status/a.json').write_text(json.dumps({'status':'ok','matrix_sha256':'old'}))
            (run/'status/a.json').write_text(json.dumps({'status':'ok','matrix_sha256':'new'}))
            result=module.audit(ref,[run])
            self.assertEqual(result['status_counts'],{'ok':1,'pending':1})
            self.assertEqual(result['problems'][0]['dataset'],'a')
            (run/'status/a.json').write_text(json.dumps({'status':'ok','matrix_sha256':'old'}))
            self.assertEqual(module.audit(ref,[run])['problems'],[])


if __name__ == '__main__': unittest.main()
