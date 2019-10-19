---
layout: archive
title: "Resume"
permalink: /resume/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

Software Development Experience
======

* Upgraded Wisconsin Plasma Physics Laboratory (WiPPL) data acquisition pipeline from the digitizers to the MDSplus data storage server to include post-processing of data from multiple sources including a MySQL server.
* Developed a customizable Python GUI for viewing data from the lab's MDSplus server to replace a 3rd party Java app.
* Wrote Python scripts to analyze GBs of data from multiple plasma shots to create profiles of relevant plasma parameters.
* Developed LabView applications for controlling power supplies, diagnostics, digitizers, and other lab equipment for running plasma experiments.

Statistical Analysis Experience
======
* Developed a Bayesian analysis technique for absolute wavelength and point-spread function calibration of a Fabry-P\'erot spectromter enabling high-precision ion temperature and velocity measurements in low-temperature plasmas.
* Analyzed data from probes by fitting multivariate regression models to determine plasma parameters such as density and temperature.
* Fit optical measurements to profiles of plasma parameters by using Bayesian inference techniques

Professional Communications
======
* Presented posters on low-temperature plasma physics research annually to a community of mainly fusion researchers at the American Physical Society Division of Plasma Physics
* Presented current research to UW-Madison plasma physics department twice a year.
* Worked with the physics department Director of Undergraduate Studies to simplify models for diagnostics so that undergraduates can fit data collected in their senior-level physics lab class.
* First author on a paper selected as Editor's Choice in Review of Scientific Instruments. Contributed to other papers published by colleagues in the lab.  

Education
======
* Ph.D in Physics, University of Wisconsin, Jan. 2020 (expected)
  * Specialized in plasma physics focusing on astrophysical phenomena in the laboratory.
* B.S. in Engineering Physics, Cornell University, 2011
  * Graduated *Magna Cum Laude*.

{% comment %}
Work experience
======
* Summer 2015: Research Assistant
  * Github University
  * Duties included: Tagging issues
  * Supervisor: Professor Git

* Fall 2015: Research Assistant
  * Github University
  * Duties included: Merging pull requests
  * Supervisor: Professor Hub
{% endcomment %}

Skills
======
* Python (Numpy, SciPy, Matplotlib, PyQt)
* Bayesian data analysis
* Advanced Mathematics (multivariable calculus, linear algebra, differential equations)
* LabView
* LaTeX
* Basic SQL
* Basic Tensorflow and scikit-learn
* Familiarity with Linux and the command line


{% comment %}
Publications
======
  <ul>{% for post in site.publications %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Talks
======
  <ul>{% for post in site.talks %}
    {% include archive-single-talk-cv.html %}
  {% endfor %}</ul>
  
Teaching
======
  <ul>{% for post in site.teaching %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Service and leadership
======
* Currently signed in to 43 different slack teams
{% endcomment %}
