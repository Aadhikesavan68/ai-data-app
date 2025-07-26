import pdfkit
from jinja2 import Environment, FileSystemLoader

def generate_report(summary, output_file="report.pdf"):
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template("report_template.html")
    html_out = template.render(summary)

    pdfkit.from_string(html_out, output_file)
