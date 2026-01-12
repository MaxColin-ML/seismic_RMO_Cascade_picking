import numpy as np

def interp_curves(curve_dict):
    curve_dict_new = dict()
    for name, curve in curve_dict.items():
        curve = curve[np.argsort(curve[:, 0]), :]
        x_new = np.arange(curve[0, 0], curve[-1, 0])
        y_new = np.interp(x_new, curve[:, 0], curve[:, 1])
        curve_interp = np.array([x_new, y_new]).T.astype(np.int32)
        curve_dict_new[name] = curve_interp
    return curve_dict_new


def get_manual_lab(m_p_dict_path):
    lab = np.load(m_p_dict_path, allow_pickle=True).item()
    for key, value in lab.items():
        lab[key] = np.array(value)
    
    # interp curves
    lab_curve = interp_curves(lab)
    return lab_curve