# should_continue = True
# if should_continue:
#     print('Hello')

known_people = ['Tina', 'Benny', 'John', 'Mary']

person = input('Enter the person you know: ')

# if person in known_people:
#     print('You know this person!')

# if person not in known_people:
#     print("You don't know this person!")

# if person in known_people:
#     print('You know this person!')
# else:
#     print("You don't know this person!")

# if person in known_people:
#     print('You know {}!'.format(person))
# else:
#     print("You don't know {}!".format(person))

# def who_do_you_know():
#     # Ask the user for a list of people they know
#     # Split the string into  a list
#     # Return that list
#     pass

# def ask_user():
#     # Ask user for a name
#     # See if their name is in the list of people they know
#     # Print out that they know the person
#     pass

def who_do_you_know():
    people = input('Enter the names of people you know, separaded by commas: ') # Ask the user for a list of people they know
    people_list = people.split(',') # Split the string into  a list

    people_without_space = []
    for person in people_list:
        people_without_space.append(person.strip())
    return people_without_space # Return that list

def ask_user():
    person = input('Enter a name of someone you know: ') # Ask user for a name
    if person in who_do_you_know():
        print('You know {}!'.format(person))

ask_user()