import sys

# Importing source code and config files from elsewhere
import tv_errortesting  # Import the tv_errortesting module

if __name__ == '__main__':
    args = sys.argv
    print(args)

    # Call a specific function from tv_errortesting.py
    # Assuming tv_errortesting.py has a function named 'main' to run
    if hasattr(tv_errortesting, 'main'):
        tv_errortesting.main(args[1:])  # Pass command-line arguments if needed
    else:
        print("tv_errortesting.py does not have a main() function to call.")

