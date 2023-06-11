import pickle

try:
    # Load the 'a.pkl' file
    with open('/home/rickleung/PycharmProjects/pythonProject/Tiny_ML_Temp/algorithm/assets/mcu_models/mcunet-5fps.pkl', 'rb') as f:
        data_a = pickle.load(f)

    # Retrieve the value associated with the key from 'a.pkl'
    value = data_a.get('key')
    if value is None:
        print("'key' does not exist in 'a.pkl'")
    else:
        # Load the 'b.pkl' file
        with open('/home/rickleung/Downloads/luis.pkl', 'rb') as f:
            data_b = pickle.load(f)

        # Add the value under the same key
        data_b['key'] = value

        # Save the updated dictionary back to 'b.pkl'
        with open('/home/rickleung/Downloads/luis.pkl', 'wb') as f:
            pickle.dump(data_b, f)

except (EOFError, pickle.UnpicklingError):
    print('An error occurred while loading or saving a pickle file')