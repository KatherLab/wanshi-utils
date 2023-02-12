from wanshi.visualizations.crona.met_brewer.palettes import met_brew

color_spam = met_brew(name="Java", brew_type="continuous")

color_Level_of_Consensus = ['red', 'orange', 'yellow']

color_Guideline_Type = ['#0bb50e', '#00ff10'] # green, light green

color_guide_line_group = color_spam[:5]

color_year = ['#878bb6', '#7b7dbd', '#696ac7','#5857d1', '#4e4bd7','#403cdf', '#342fe6', '#2620ee','#160ff7', '#0800ff']

color_link = 'purple'

if __name__ == '__main__':
    assert len(color_Level_of_Consensus) == 3
    assert len(color_Guideline_Type) == 2
    assert len(color_guide_line_group) == 5
