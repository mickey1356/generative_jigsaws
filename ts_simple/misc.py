import math

def C(value, epoch: int, global_step: int, interpolation="linear") -> float:
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        if not isinstance(value, list):
            raise TypeError("Scalar specification only supports list, got", type(value))
        if len(value) == 3:
            value = [0] + value
        if len(value) >= 6:
            select_i = 3
            for i in range(3, len(value) - 2, 2):
                if global_step >= value[i]:
                    select_i = i + 2
            if select_i != 3:
                start_value, start_step = value[select_i - 3], value[select_i - 2]
            else:
                start_step, start_value = value[:2]
            end_value, end_step = value[select_i - 1], value[select_i]
            value = [start_step, start_value, end_value, end_step]
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        if isinstance(end_step, int):
            current_step = global_step
        elif isinstance(end_step, float):
            current_step = epoch
        t = max(min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0)
        if interpolation == "linear":
            value = start_value + (end_value - start_value) * t
        elif interpolation == "exp":
            value = math.exp(math.log(start_value) * (1 - t) + math.log(end_value) * t)
        else:
            raise ValueError(
                f"Unknown interpolation method: {interpolation}, only support linear and exp"
            )
    return value
