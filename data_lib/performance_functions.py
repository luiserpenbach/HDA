
g0 = 9.81

def calc_isp(avg_thrust_N, avg_mdot_kg_s):
    # Specific Impulse (s)
    isp = avg_thrust_N / (avg_mdot_kg_s * g0)
    return isp

def calc_cstar(throat_area_m2):
    pass

def calc_cf():
    pass


if __name__ == '__main__':
    # Test functions
    isp_demo = calc_isp(1900, 1.05)
    print(isp_demo)