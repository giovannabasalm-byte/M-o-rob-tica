import winsound

sons = [
    "pianinho/Sons/do.wav",
    "pianinho/Sons/re.wav",
    "pianinho/Sons/mi.wav",
    "pianinho/Sons/fa.wav",
    "pianinho/Sons/sol.wav"
]

for som in sons:

    print("Tocando:", som)

    winsound.PlaySound(
        som,
        winsound.SND_FILENAME
    )