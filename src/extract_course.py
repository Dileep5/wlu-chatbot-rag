import re

text = """
CP600Practical Algorithm Design0.5 Credit
The techniques of algorithm design form one of the core practical technologies of computer science. This course focuses on advanced techniques for designing and analysing algorithms.
"""

course_match = re.search(r"(CP\d{3})", text)

credit_match = re.search(r"(\d+\.\d+)\s*Credit", text)

course_code = course_match.group(1)
credits = credit_match.group(1)

start = course_match.end()
end = credit_match.start()

title = text[start:end].strip()

description = text[credit_match.end():].strip()

print("Course Code:", course_code)
print("Title:", title)
print("Credits:", credits)
print("Description:", description)