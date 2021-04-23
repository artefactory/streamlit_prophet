def reverse_list(l):
    reversed_list = [l[len(l) - 1 - i] for i, x in enumerate(l)]
    return reversed_list


div10_bluepink = ["#052d57", "#0e447d", "#175aa3", "#8badd1", "#c5d6e8",
                  "#fbd7ea", "#f7afd5", "#ee5eab", "#e5418a", "#dc2469"]

div9_bluepink = ["#052d57", "#175aa3", "#6e98c6", "#c5d6e8", "#ffffff",
                 "#fbd7ea", "#f7afd5", "#ee5eab", "#dc2469"]

div8_bluepink_dark = ["#0c274c", "#052d57", "#024189", "#175aa3", "#ee5eab",
                      "#ea4496", "#dc2469", "#b00c67"]

div9_bluepink_dark = ["#0c274c", "#052d57", "#024189", "#175aa3", "#a3a3a3",
                      "#ee5eab", "#ea4496", "#dc2469", "#b00c67"]

seq10_green_r = ["#337788", "#3a8291", "#408d99", "#4da2aa", "#57b2b7",
                 "#60c2c4", "#66cccc", "#81d5d5", "#a9e2e2", "#d1f0f0"]
seq10_green = reverse_list(seq10_green_r)

seq10_blue_r = ["#0c274c", "#052d57", "#153872", "#024189", "#00499a",
                "#175aa3", "#2c69ab", "#3f77b3", "#5083ba", "#608ec0"]
seq10_blue = reverse_list(seq10_blue_r)

seq10_pink_r = ["#b00c67", "#c61868", "#dc2469", "#e53886", "#ea4496",
                "#f06db3", "#f286c0", "#f49bcb", "#f6acd4", "#f9c8e2"]
seq10_pink = reverse_list(seq10_pink_r)

cat10_strong = ["#002244", "#ff0066", "#66cccc", "#ff9933", "#337788",
                "#429e79", "#474747", "#f7d126", "#ee5eab", "#b8b8b8"]
