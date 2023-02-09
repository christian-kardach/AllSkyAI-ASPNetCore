using Microsoft.AspNetCore.Mvc;

namespace AllSkyAI_ASPNetCore.Controllers
{
    public class HomeController : Controller
    {
        Configuration configuration = new Configuration();
        public IActionResult Index()
        {
            ViewBag.url = configuration.Url;
            return View("live");
        }
    }
}
