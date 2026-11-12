import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'analysis'))
from sptrsv_matrix_contract import validate_tc_factor

with tempfile.TemporaryDirectory() as tmp:
    f = Path(tmp) / 'm.mtx'
    def valid_write(entries):
        f.write_text('%%MatrixMarket matrix coordinate real general\n' + f'16 16 {len(entries)}\n' + ''.join(f'{i} {j} {v}\n' for i,j,v in entries))
    valid = [(i,i,1) for i in range(1,17)]
    valid_write(valid); assert validate_tc_factor(f) == {'rows':16,'nnz':16}
    for invalid in (valid[:-1], valid[:4]+[(5,5,2)]+valid[5:], valid+[(1,1,1)], valid[:3]+[(4,5,1)]+valid[3:]):
        valid_write(invalid)
        try:
            validate_tc_factor(f)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted an invalid factor')
