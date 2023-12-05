namespace AllSkyAI_ASPNetCore
{
    public class LabelMap
    {
        public static readonly string[] Labels = new[] {"clear",
                                                        "heavy_clouds",
                                                        "light_clouds",
                                                        "not_running",
                                                        "precipitation"};

        public static string[] readLabels(string labelsPath)
        {
            List<string> _labels = new List<string>();

            var lines = File.ReadLines(labelsPath);
            foreach (var line in lines)
            {
                _labels.Add(line.Trim());
            }
            return _labels.ToArray();
        }
    }
}
