from dataclasses import dataclass, field

@dataclass
class Person:
    name: str
    age: int


p = Person('Ben', 26)
