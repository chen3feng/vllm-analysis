# Jekyll plugin: rewrite local submodule source links to GitHub blob URLs.
# Local VS Code:   links stay as   ../vllm/vllm/path.py#L123
# GitHub Pages:    links become    https://github.com/vllm-project/vllm/blob/<commit>/vllm/path.py#L123

Jekyll::Hooks.register :site, :post_render do |site|
  base = site.config.dig("source_rewrite", "base")
  next unless base

  site.pages.each do |page|
    next unless page.output

    page.output = page.output.gsub(%r{href="\.\./vllm/vllm/([^"]+)"}) do
      "href=\"#{base}/vllm/#{$1}\""
    end
    page.output = page.output.gsub(%r{href="\.\./csrc/([^"]+)"}) do
      "href=\"#{base}/csrc/#{$1}\""
    end
    page.output = page.output.gsub(%r{href="\.\./(setup\.py[^"]*)"}) do
      "href=\"#{base}/#{$1}\""
    end
    page.output = page.output.gsub(%r{href="\.\./(CMakeLists\.txt[^"]*)"}) do
      "href=\"#{base}/#{$1}\""
    end
    page.output = page.output.gsub(%r{href="\.\./(pyproject\.toml[^"]*)"}) do
      "href=\"#{base}/#{$1}\""
    end
    page.output = page.output.gsub(%r{href="\.\./(cmake/[^"]+)"}) do
      "href=\"#{base}/#{$1}\""
    end
  end
end
