def generate_negative_offset_array_from_df(df_filtered):
    return [[-0.75, -0.75, -0.75, -0.75] for _ in range(len(df_filtered))]

def generate_negative_offset_array_from_df_abbott(df_filtered):
    return [[-0.75] * 8 for _ in range(len(df_filtered))]


def generate_amplitude_matrix(df_filtered):
    df_filtered = df_filtered.reset_index(drop=True)

    amplitude_matrix = [[0.0, 0.0, 0.0, 0.0] for _ in range(df_filtered.shape[0])]

    for idx, row in df_filtered.iterrows():
        contact = row['contact']
        amplitude = float(row['amplitude'])
        hemisphere = row["hemisphere"]

        if hemisphere == 'Right' and contact in ["0", "1", "2", "3"]:
            amplitude_matrix[idx][int(contact)] = amplitude
        elif hemisphere == 'Left' and contact in ["4", "5", "6", "7"]:
            contact_index = int(contact) - 4  # map 4 -> 0, 5 -> 1, etc.
            amplitude_matrix[idx][contact_index] = amplitude

    return amplitude_matrix


def generate_amplitude_matrix_abbott(df_filtered):
    df_filtered = df_filtered.reset_index(drop=True)
    amplitude_matrix = [[0.0] * 8 for _ in range(df_filtered.shape[0])]
    mapping = {
        'R': {
            '1': [0], '2A': [1], '2B': [2], '2C': [3],
            '3A': [4], '3B': [5], '3C': [6], '4': [7],
            '2ABC': [1, 2, 3], '2AB': [1, 2], '2AC': [1, 3], '2BC': [2, 3],
            '3ABC': [4, 5, 6], '3AB': [4, 5], '3AC': [4, 6], '3BC': [5, 6],
        },
        'L': {
            '9': [0], '10A': [1], '10B': [2], '10C': [3],
            '11A': [4], '11B': [5], '11C': [6], '12': [7],
            '10ABC': [1, 2, 3], '10AB': [1, 2], '10AC': [1, 3], '10BC': [2, 3],
            '11ABC': [4, 5, 6], '11AB': [4, 5], '11AC': [4, 6], '11BC': [5, 6],
        }
    }
    for idx, row in df_filtered.iterrows():
        contact = row['contact']
        amplitude = float(row['amplitude'])
        hem = row['hemisphere'].upper()[0]
        if hem in mapping and contact in mapping[hem]:
            for i in mapping[hem][contact]:
                amplitude_matrix[idx][i] = amplitude
        else:
            print(f"[Abbott] Unknown contact '{contact}' at row {idx}")
    return amplitude_matrix