# grade1 = 77
# grade2 = 80
# grade3 = 60

# print((grade1 + grade2 + grade3) / 3)

grades = [77, 80, 60]
# print(sum(grades) / len(grades))

tuple_grades = (3, 4, 5, 6) # immutable
set_grades = {70, 80, 90, 100, 100} # unique & unordered
# print(tuple_grades)
# print(set_grades)
# tuple_grades = tuple_grades + (100, )
# print(type((100, )))
# print(tuple_grades)

# print(grades[0])

# grades[0] = 123
# print(grades)
# print(grades[:2])
# print(grades[-1])

# print(tuple_grades[0])
# tuple_grades[0] = 87

# set_grades.add(10)
# print(set_grades)

your_lottery_numbers = {1, 2, 3, 4, 5}
winning_numbers = {1, 3, 5, 7, 9, 11}
print(your_lottery_numbers.intersection(winning_numbers))
print(your_lottery_numbers.union(winning_numbers))
print({1, 2, 3, 4}.difference({1, 2}))