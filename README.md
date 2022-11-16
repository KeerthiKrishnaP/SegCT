# Opimization_of_resin_infusion_process
Composite materials, have better specific properties than many traditional materials. But despite their promised advantages, industry has been slow to apply composites as they wait to see if an automatable composites manufacturing solution with high properties will become available. Resin infusion processing is being developed in many laboratories in an attempt to find this suitable process. Qualification of an optimized resin infusion process will require significant performance data and accurate manufacturing simulation tools. This is currently hindered by a lack of standardized methods for part characterization, and a lack of understanding of the different flow phenomena that are involved in the simulation of flow processing. This work aims to help meet these needs, so as to allow further implementation of fiber-reinforced composite materials in industry. To improve simulation capabilities, individual modeling systems for each of the following flow phenomena are developed: permeability, compressibility, dynamic viscosity, and dual-scale flow. These models were developed based on relationships previously proposed in the literature. 

# IDE Configuration 
It is highly recommended to be working with VSCode, an IDE that does not need to be presented. Internally, we use a set of code extensions enabling a minimum of code standardization, making the life of many developers more enjoyable. Those extensions can be downloaded directly via the VSCode extension store. This goes hand and hand with properly configured VSCode workspace settings.


# Good code
## Return Statements
A comment that will come back often in PR reviews is the spacing in your code. The overall strategy is to split your code by functional blocks, aka adding empty lines to differentiate loops, if-statements or clusters of similar actions. There are also a few more guidelines:

Return statements should be isolated from any code blocks
Do not use spacing between a function name and the first line of code
An application of those guidelines is illustrated below:

```python
#do
def function():
  return object

#don't
def function():

  return object

#do
def function():
  object = get_object()

  return object

don't
def function():
  object = get_object()
  return object
```
  
## Naming conventions
"There are only two hard things in Computer Science: cache invalidation and naming things." - Phil Karlton
That is exactly why it is important everyone follow guidelines regarding naming conventions, especially when moving quickly as a team. Here are a set of rules that will most likely guide you through any problem you would face:

## General
Do not use abbreviations
Use at least 2 words for function names
Boolean variables should be infered from their name (e.g. start with is_ or has_)

## Python
```
Use snake_case for folder names, function names
Use PascalCase for class names
Use SCREAMING_SNAKE_CASE for constants
Use _ prefix for private functions
```

## Typing
Typing is key to maintainability. It will increase the readability of the code, but will also passively document your code. Finally, type checking will help to find some obvious bugs.

# Github
## Branches:
We have a simple convention for branch naming: ```{initials}/{descriptive-kebab-case}```. Keep them all lowercase. For John Doe working on a feature A, that would be jd/feature-a.

## Commits:
The Conventional Commits specification is a lightweight convention on top of commit messages. It provides an easy set of rules for creating an explicit commit history; which makes it easier to write automated tools on top of. This convention dovetails with SemVer, by describing the features, fixes, and breaking changes made in commit messages. Learn more here.

## Pull requests:
There are simple rules in regards to our PR management:

## Link your PRs to their related Notion tickets;
Use prefixes for your PR titles among [FIX], [FEAT], [REFACTOR], [RELEASE], [HOTFIX], [TEST];
If your code affects the application build, be sure to update the README.md;
Do not merge a PR until all comments are resolved;
Remove your branch after merging;

# Get Started
1 - install our Git hooks
```
yarn husky:install
yarn husky:prepare
```
2 - install all dependencies
```
make hard-install
```
