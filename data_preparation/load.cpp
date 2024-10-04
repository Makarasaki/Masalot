#include <iostream>
#include <fstream>
#include <vector>
#include <string>

class Person {
public:
    int age;
    std::string name;

    // Constructor
    Person() : age(0), name("") {}
    Person(int a, const std::string& n) : age(a), name(n) {}

    // Method to load the list of objects from a file
    static std::vector<Person> loadFromFile(const std::string& filename) {
        std::vector<Person> persons;
        std::ifstream inFile(filename, std::ios::in | std::ios::binary);
        if (!inFile) {
            std::cerr << "Error opening file for reading" << std::endl;
            return persons;
        }
        size_t size;
        inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
        persons.resize(size);
        for (auto& person : persons) {
            size_t nameLength;
            inFile.read(reinterpret_cast<char*>(&person.age), sizeof(person.age));
            inFile.read(reinterpret_cast<char*>(&nameLength), sizeof(nameLength));
            person.name.resize(nameLength);
            inFile.read(&person.name[0], nameLength);
        }
        inFile.close();
        return persons;
    }
};

int main() {
    // Load the list of Person objects from a file
    std::vector<Person> persons = Person::loadFromFile("persons.dat");

    // Print the attributes of each Person object
    for (const auto& person : persons) {
        std::cout << "Person name: " << person.name << ", age: " << person.age << std::endl;
    }

    return 0;
}
