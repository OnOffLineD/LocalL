fn main() {
    #[cfg(windows)]
    {
        let mut res = winres::WindowsResource::new();
        res.set_icon("assets/icon.ico");
        res.set("ProductName", "LocalL");
        res.set("FileDescription", "LocalL Fast AI Search");
        let _ = res.compile();
    }
}
