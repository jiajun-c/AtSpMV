This experiment is invalid for performance claims. The experimental device-vector
coefficient path changed the final temperature relative to the original coefficient
formula (relative max difference about 0.0077). The full-field validator rejected it.
Mass conservation and energy decay alone did not detect this mismatch. The current
implementation uses a single synchronized HostReadWrite view for a bulk scalar
coefficient update, and requires a fresh paired full-field validation.
