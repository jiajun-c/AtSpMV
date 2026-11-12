"""Matrix Market contract used by the aligned TC-SpTRSV comparison."""
import math


def validate_tc_factor(path):
    """Require sorted, unique lower-triangular real CSR-equivalent triplets.

    The TC artifact replaces diagonal values with 1 and pads dimensions to 16.
    Reject inputs for which this would alter the operator, rather than reporting
    incomparable timings. Explicit zero off-diagonal entries are also rejected.
    """
    with open(path, encoding='utf-8') as f:
        header = f.readline().strip().lower().split()
        if header != ['%%matrixmarket', 'matrix', 'coordinate', 'real', 'general']:
            raise ValueError('expected real general Matrix Market coordinate format')
        line = f.readline()
        while line and (not line.strip() or line.lstrip().startswith('%')):
            line = f.readline()
        try:
            n, m, nnz = map(int, line.split())
        except (ValueError, TypeError):
            raise ValueError('invalid Matrix Market dimensions') from None
        if n <= 0 or n != m or n % 16 or nnz < n:
            raise ValueError('factor must be square, nonempty and padded to 16')
        previous = (0, 0)
        diagonals = 0
        seen = 0
        for line in f:
            if not line.strip() or line.lstrip().startswith('%'):
                continue
            fields = line.split()
            if len(fields) != 3:
                raise ValueError('invalid coordinate')
            try:
                i, j = int(fields[0]), int(fields[1]); value = float(fields[2])
            except ValueError:
                raise ValueError('invalid coordinate value') from None
            if not (1 <= j <= i <= n) or (i, j) <= previous:
                raise ValueError('coordinates must be sorted, unique and lower triangular')
            if not math.isfinite(value) or value == 0:
                raise ValueError('factor entries must be finite and nonzero')
            if i == j:
                if value != 1.0:
                    raise ValueError('TC reference overwrites a non-unit diagonal')
                diagonals += 1
            previous = (i, j)
            seen += 1
        if seen != nnz or diagonals != n:
            raise ValueError('coordinate count mismatch or missing diagonal')
        return {'rows': n, 'nnz': nnz}
