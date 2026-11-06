def calculate_density(people_count):

    if people_count <= 20:
        return "LOW"

    elif people_count <= 50:
        return "MEDIUM"

    else:
        return "HIGH"