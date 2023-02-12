from wanshi.visualizations.crona.met_brewer.palettes import met_brew

color_spam = met_brew(name="Java", brew_type="continuous")

color_Level_of_Consensus = ['red', 'orange', 'yellow']

color_Guideline_Type = ['#0bb50e', '#00ff10'] # green, light greem

color_guide_line_group = color_spam[:5]

color_link = 'purple'

if __name__ == '__main__':
    assert len(color_Level_of_Consensus) == 3
    assert len(color_Guideline_Type) == 2
    assert len(color_guide_line_group) == 5
