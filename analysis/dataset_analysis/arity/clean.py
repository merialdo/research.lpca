def clean_entity_bit(bit):
    clean_bit = bit[1:-1]
    clean_bit = clean_bit.replace("http://rdf.freebase.com/ns", "")

    if clean_bit.startswith("/m."):
        clean_bit = "/m/" + clean_bit[3:]

    return clean_bit

def clean_relation_bit(bit):
    clean_bit = bit[1:-1]
    clean_bit = clean_bit.replace("http://rdf.freebase.com/ns", "")
    clean_bit = clean_bit.replace(".", "/")
    return clean_bit

def clean_line(line):
    line = line.strip()
    line = line[:-1]    # delete "."
    line = line.strip()

    bits = line.split("\t")

    return [clean_entity_bit(bits[0]), clean_relation_bit(bits[1]), clean_entity_bit(bits[2])]

entities = set()
with open("entities.txt", "r") as input_file:
    lines = input_file.readlines()
    for line in lines:
        line = line.strip()
        if len(line) != 0:
            entities.add(line)

relations = set()
with open("relations.txt", "r") as input_file:
    lines = input_file.readlines()
    for line in lines:
        line = line.strip()
        if len(line) != 0:
            relations.add(line)

with open("freebase-rdf-latest", "r") as freebase_input:
    with open("freebase-clean", "w") as freebase_output:

        buffer = []
        input_line_count = 0
        for line in freebase_input:
            input_line_count += 1
            if input_line_count % 10000000 == 0:
                print("Read 10M lines...")

            if "http://www.w3.org" in line:
                continue

            if "wiki" in line:
                continue

            clean_bits = clean_line(line)

            keep = clean_bits[0] in entities or clean_bits[2] in entities
            keep = keep and clean_bits[1] in relations

            if keep:
                buffer.append("\t".join(clean_bits) + "\n")

            if len(buffer) > 10000000:
                print("Writing buffer of size 10M...")
                freebase_output.writelines(buffer)
                buffer = []

                freebase_output.writelines(buffer)
        buffer = []
