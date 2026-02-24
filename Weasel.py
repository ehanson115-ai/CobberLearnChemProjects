# WEASEL EVOLUTION PROGRAM WITH FITNESS PLOT + USER INPUT
import random
import string
import matplotlib.pyplot as plt
import os

# ---- USER INPUT ----
TARGET = input("Enter the phrase you want to evolve: ").upper()

ALLOWED_CHARS = string.ascii_uppercase + " "


# ---- FITNESS FUNCTION ----
def score_string(candidate):
    return sum(1 for c, t in zip(candidate, TARGET) if c == t)


# ---- MUTATION FUNCTION ----
def mutate_string(parent, mutation_rate=0.05):
    new_string = ""
    for char in parent:
        if random.random() < mutation_rate:
            new_string += random.choice(ALLOWED_CHARS)
        else:
            new_string += char
    return new_string


# ---- RANDOM STRING GENERATOR ----
def random_string(length):
    return ''.join(random.choice(ALLOWED_CHARS) for _ in range(length))


# ---- EVOLUTION LOOP ----
parent = random_string(len(TARGET))
parent_score = score_string(parent)

generation = 0
fitness_over_time = []

while parent_score < len(TARGET):
    generation += 1

    offspring_list = [mutate_string(parent) for _ in range(100)]
    best_offspring = max(offspring_list, key=score_string)
    best_score = score_string(best_offspring)

    if best_score > parent_score:
        parent = best_offspring
        parent_score = best_score

    fitness_over_time.append(parent_score)

    print(f"Generation {generation}: Score = {parent_score}, String = '{parent}'")

print("\nTarget reached!")

# ---- PLOT FITNESS OVER TIME ----
plt.figure()
plt.plot(fitness_over_time)
plt.xlabel("Generation")
plt.ylabel("Fitness Score")
plt.title("Fitness Over Time (Weasel Evolution)")

# ---- SAVE TO WEASEL DIRECTORY ----
save_dir = "/mnt/data/Weasel"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "fitness_plot.png")

plt.savefig(save_path)
plt.show()

print(f"\nPlot saved to: {save_path}")


