from wanshi.visualizations.crona.met_brewer.palettes import met_brew

color_spam = met_brew(name="Hokusai3", n = 5, brew_type="continuous")
color_spam2 = met_brew(name="Hiroshige", n = 10, brew_type="continuous")
color_spam3 = met_brew(name="Hiroshige", n = 5, brew_type="continuous")
color_spam4 = met_brew(name="Hokusai2", n = 9, brew_type="continuous")

color_Level_of_Consensus = [color_spam[2], color_spam[1], color_spam[0]]

color_Guideline_Type = [color_spam2[2], color_spam2[4]] # green, light green

color_guide_line_group = color_spam3[:5]
print(color_guide_line_group)
color_year = color_spam4

color_link = 'grey'

gap = 0.1

font_size = 9

alphaY = 0.1
alphaP = 0.03


if __name__ == '__main__':
    assert len(color_Level_of_Consensus) == 3
    assert len(color_Guideline_Type) == 2
    assert len(color_guide_line_group) == 5
