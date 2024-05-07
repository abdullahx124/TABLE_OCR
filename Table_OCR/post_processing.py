def add_serial_number_with_header_ACPL(data):
    header_row = data[0]
    # Adding "SR" header to the first row
    header_row.insert(0, "SR")

    data_with_serial = [header_row]  # Initialize with the modified header row

    # Iterating through rows starting from the second row
    for index, row_data in enumerate(data[1:], start=1):
        row_data.insert(0, f"{index}")
        data_with_serial.append(row_data)

    return data_with_serial
