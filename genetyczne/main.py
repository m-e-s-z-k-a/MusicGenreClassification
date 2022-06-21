import click
from datetime import datetime
from typing import List, Dict
from midiutil import MIDIFile
from pyo import *

from genetic import genome_generator, Genome, pair_selection, sp_crossover, mutation

NOTE_BITS = 4
KEYS = ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"]
SCALES = ["major", "minorM", "dorian", "phrygian", "lydian", "mixolydian", "majorBlues", "minorBlues"]

def bits_to_int(bits):
    retval = 0
    for i in range(0, len(bits), 1):
        retval += bits[i] * pow(2, i)
    return retval

def genome_to_melody(genome, bars_num, notes_num, steps_num, pauses, key, scale, root):
    notes = [genome[i * NOTE_BITS:i * NOTE_BITS + NOTE_BITS] for i in range(bars_num * notes_num)]

    note_len = 4 / float(notes_num)

    scl = EventScale(root=key, scale=scale, first=root)

    melody = {"notes": [], "velocity": [], "beat": []}
    for note in notes:
        num = bits_to_int(note)

        if not pauses:
            num = int(num % pow(2, NOTE_BITS - 1))

        if num >= pow(2, NOTE_BITS - 1):
            melody["notes"] += [0]
            melody["velocity"] += [0]
            melody["beat"] += [note_len]

        else:
            if len(melody["notes"]) > 0 and melody["notes"][-1] == num:
                melody["beat"][-1] += note_len
            else:
                melody["notes"] += [num]
                melody["velocity"] += [127]
                melody["beat"] += [note_len]

    steps = []
    for step in range(0, steps_num, 1):
        steps.append([scl[int((note+2*step) % len(scl))] for note in melody["notes"]])

    melody["notes"] = steps
    return melody


def event_from_genome(genome, bars_num, notes_num, steps_num, pauses, key, scale, root, bpm):
    melody = genome_to_melody(genome, bars_num, notes_num, steps_num, pauses, key, scale, root)
    event_array =  []
    for i in range(0, len(melody["notes"]), 1):
        event_array.append(Events(midinote = EventSeq(melody["notes"][i], occurrences=1),
                                    midivel = EventSeq(melody["velocity"], occurrences = 1),
                                    beat = EventSeq(melody["beat"], occurrences = 1),
                                    attack = 0.001, decay = 0.05, sustain = 0.5, release = 0.005, bpm = bpm))
    return event_array


def metronome(bpm):
    m = Metro(time=1 / (bpm / 60.0)).play()
    t = CosTable([(0, 0), (50, 1), (200, .3), (500, 0)])
    amp = TrigEnv(m, table=t, dur=.25, mul=1)
    f = Iter(m, choice=[660, 440, 440, 440])
    return Sine(freq=f, mul=amp).mix(2).out()


def fitness(genome, serv, bars_num, notes_num, steps_num, pauses, key, scale, root, bpm):
    m = metronome(bpm)
    events = event_from_genome(genome, bars_num, steps_num, notes_num, pauses, key, scale, root, bpm)
    for i in range(0, len(events), 1):
        events[i].play()
    serv.start()

    rating = input("Rate the melody (0 - 5)")

    for i in range(0, len(events), 1):
        events[i].stop()
    serv.stop()
    time.sleep(1)

    rating = int(rating)
    return rating

def get_midi(filename, genome, bars_num, notes_num, steps_num, pauses, key, scale, root, bpm):
    melody = genome_to_melody(genome, bars_num, notes_num, steps_num, pauses, key, scale, root)

    if len(melody["notes"][0]) != len(melody["beat"]) or len(melody["notes"][0]) != len(melody["velocity"]):
        raise ValueError("cos jest nie tak!")

    midi_file = MIDIFile(1)

    track = 0
    channel = 0

    time = 0.0
    midi_file.addTrackName(track, time, "Sample Track")
    midi_file.addTempo(track, time, bpm)

    for i in range(0, len(melody["velocity"]), 1):
        if melody["velocity"][i] > 0:
            for j in range(0, len(melody["notes"]), 1):
                midi_file.addNote(track, channel, melody["notes"][j], time, melody["beat"][i], melody["velocity"][i])
        time += melody["beat"][i]

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            midi_file.writeFile(f)


@click.command()
@click.option("--bars-num", default=8, prompt='Number of bars:', type=int)
@click.option("--notes-num", default=4, prompt='Notes per bar:', type=int)
@click.option("--steps-num", default=1, prompt='Number of steps:', type=int)
@click.option("--pauses", default=False, prompt='Introduce Pauses?', type=bool)
@click.option("--key", default="C", prompt='Key:', type=click.Choice(KEYS, case_sensitive=False))
@click.option("--scale", default="major", prompt='Scale:', type=click.Choice(SCALES, case_sensitive=False))
@click.option("--root", default=4, prompt='Scale Root:', type=int)
@click.option("--population-size", default=10, prompt='Population size:', type=int)
@click.option("--mutations-num", default=2, prompt='Number of mutations:', type=int)
@click.option("--mutations-prob", default=0.5, prompt='Mutations probability:', type=float)
@click.option("--bpm", default=128, type=int)
def main(bars_num, notes_num, steps_num, pauses, key, scale, root, population_size, mutations_num, mutations_prob, bpm):
    folder = str(int(datetime.now().timestamp()))
    population = [genome_generator(bars_num * notes_num * NOTE_BITS) for _ in range(population_size)]
    serv = Server(audio="jack")
    serv.boot().start()

    #serv = Server().boot()
    population_id = 0

    running = True
    while running:
        random.shuffle(population)
        fitness_population = [(genome, fitness(genome, serv, bars_num, notes_num, steps_num, pauses, key, scale, root, bpm)) for genome in population]
        population_fitness_sorted = sorted(fitness_population, key=lambda e: e[1], reverse=True)

        population = [e[0] for e in population_fitness_sorted]
        next_gen = population[0:2]

        for j in range(int(len(population) / 2) - 1):

            def fitness_lookup(genome):
                for e in fitness_population:
                    if e[0] == genome:
                        return e[1]
                return 0

            parents = pair_selection(population, fitness_lookup)
            child1, child2 = sp_crossover(parents[0], parents[1])
            child1 = mutation(child1, num=mutations_num, probability=mutations_prob)
            child2 = mutation(child2, num=mutations_num, probability=mutations_prob)
            next_gen += [child1, child2]

    print(f"population ", population_id, " done")

    events = event_from_genome(population[0], bars_num, notes_num, steps_num, pauses, key, scale, root, bpm)
    for e in events:
        e.play()
    serv.start()
    input("here is the no1 hit …")
    serv.stop()
    for e in events:
        e.stop()

    time.sleep(1)

    events = event_from_genome(population[1], bars_num, notes_num, steps_num, pauses, key, scale, root, bpm)
    for e in events:
        e.play()
    serv.start()
    input("here is the second best …")
    serv.stop()
    for e in events:
        e.stop()

    time.sleep(1)

    print("saving population midi …")
    for i, genome in enumerate(population):
        get_midi(f"{folder}/{population_id}/{scale}-{key}-{i}.mid", genome, bars_num, notes_num, steps_num,
                            pauses, key, scale, root, bpm)
    print("done")

    running = input("continue? [Y/n]") != "n"
    population = next_gen
    population_id += 1

if __name__ == '__main__':
    main()