from beam.orchestration.units import K8SUnits

def test_k8s_units_with_limits():
    scenarios = [
        ("Memory (1.6 as string)", "1.6", "memory"),
        ("Memory (1.6 as float)", 1.6, "memory"),
        ("Memory (2 as string)", "2", "memory"),
        ("CPU (2 as string)", "2", "cpu"),
        ("CPU (2.0 as float)", 2.0, "cpu"),
        ("CPU (0.2 as float)", 0.2, "cpu"),
        ("CPU (0.2 as string)", "0.2", "cpu"),
        ("Memory (0.2Gi)", "0.2Gi", "memory"),
        ("Memory (512Gi, valid limit)", "512", "memory"),
        ("CPU (24 cores, valid limit)", "24", "cpu"),
        ("Memory (513Gi, exceeding limit)", "513", "memory"),
        ("CPU (25 cores, exceeding limit)", "25", "cpu"),
        ("Memory (600000Mi, exceeding limit)", "600000Mi", "memory"),
    ]

    for description, value, resource_type in scenarios:
        try:
            result = K8SUnits(value, resource_type=resource_type).as_str
            print(f"{description}: Success -> {result}")
        except ValueError as e:
            print(f"{description}: Error -> {str(e)}")


# Run the function with the added limit test scenarios
test_k8s_units_with_limits()