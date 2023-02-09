using Microsoft.AspNetCore.Mvc;

namespace AllSkyAI_ASPNetCore.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class ClassifyController : ControllerBase
    {
        OnnxClassify classify = new OnnxClassify();
        Configuration configuration = new Configuration();

        private readonly ILogger _logger;

        public ClassifyController(ILogger<ClassifyController> logger)
        {
            _logger = logger;
        }
        
        [HttpGet(Name = "classify")]
        public string Get()
        {
            _logger.Log(LogLevel.Information, "");
            Task<string> task3 = Task<string>.Factory.StartNew(() =>
            {
                string result = classify.ClassifyImage(configuration.Url);
                return result;
            });

            var res = task3.Result;
            return res;
            
        }
    }
}
