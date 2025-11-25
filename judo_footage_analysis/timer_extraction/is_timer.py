import re


def is_timer(input_string):
    # Define the regular expression pattern for the timer format
    pattern = re.compile(r"^\d{1,2}\s*:\s*\d{2}$")

    # Check if the input string matches the pattern
    if not pattern.match(input_string):
        return (False, None, None)

    # Split the string into minutes and seconds
    minutes, seconds = map(int, input_string.split(":"))

    # Check if seconds are within the valid range (0 to 59)
    if 0 <= seconds <= 59:
        return (True, minutes, seconds)
    else:
        return (False, None, None)


# Test cases
print(is_timer("4:00"))  # True
print(is_timer("3:80"))  # False
print(is_timer("3:20"))  # True
print(is_timer("3: 20"))  # True
print(is_timer("3 : 20"))  # True
print(is_timer("0:00"))  # True
print(is_timer("1:60"))  # False
print(is_timer("iauzhduh"))  # False
