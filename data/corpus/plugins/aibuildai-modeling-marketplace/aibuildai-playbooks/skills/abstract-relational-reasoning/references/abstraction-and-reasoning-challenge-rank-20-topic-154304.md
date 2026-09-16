# 21st place solution

Competition: abstraction-and-reasoning-challenge
Rank: #20
Source: https://www.kaggle.com/c/abstraction-and-reasoning-challenge/discussion/154304

First of all thank you so much François Chollet and kaggle team for organizing this awesome competition.
It has been really fun and creative competition for me. I have been getting score 1 for a month lol.

My solution was similar to DSL methods we have been discussed.
All the training code and test/evaluation code was processed by python methods which has the following interfaces.

- def converter(inp, objects, c) returns objects
  - inp: The original input from the dataset
  - objects: processed input data representation with some metadata. eg) rectangle info
  - c: Context object which is shared between training and testing. eg) Colors observed only in output. output shape information.

- def merger(objects, c):
  - objects: processed input data
  - c: the context object

I have more than 100 converter methods and 10 merger methods defined.

For example one task might be solved by
- [converter] identify_objects
  - Convert input array into rectangle pieces as objects
- [converter] color_with_out_color
  - Color all the found objects with color which appears only in output
- [merger] simple_merge
  - Merge all the object into one array.


I have rotation, fill, flip, move, draw_line and many types of converters and or_merge, and_merge, select_biggest_object and more mergers.

That's about it.
I'm looking forward to seeing more creative ideas and solutions than mine :)

One more note for the first month, CNN based image generation model. But it didn't work as we have so few training data.

## Real code for identify objects
```
def identify_rec2(y, x, a, visited, c, rect):

    if visited[y + 1][x + 1]:
        return
    visited[y + 1][x + 1] = True
    if a[y][x] == c.bg_color:
        return
    if y &lt; rect.y1:
        rect.y1 = y
    if y &gt; rect.y2:
        rect.y2 = y
    if x &lt; rect.x1:
        rect.x1 = x
    if x &gt; rect.x2:
        rect.x2 = x
    identify_rec2(y + 1, x + 1, a, visited, c, rect)
    identify_rec2(y + 1, x, a, visited, c, rect)
    identify_rec2(y, x + 1, a, visited, c, rect)   
    identify_rec2(y - 1, x - 1, a, visited, c, rect)
    identify_rec2(y + 1, x - 1, a, visited, c, rect)
    identify_rec2(y - 1, x + 1, a, visited, c, rect)    
    identify_rec2(y - 1, x, a, visited, c, rect)
    identify_rec2(y, x - 1, a, visited, c, rect)        


def identify_greedy(inp, objects, c):
    if len(objects) != 1:
        raise SkipException                    
        return objects
    a = objects[0].data
    height, width = a.shape
    if height &gt; 30 and width &gt; 30:
        raise SkipException                    
        return objects
    visited = init_visited(height, width)
    objs = []
    for y in range(height):
        for x in range(width):
            if visited[y + 1][x + 1]:
                continue
            if a[y][x] == c.bg_color:
                visited[y + 1][x + 1] = True
                continue
            if not visited[y + 1][x + 1]:
                rec = Rect(y,y, x, x)
                identify_rec2(y, x, a, visited, c, rec)
                obj = Object(a)
                obj.rectangle = (rec.y1, rec.y2, rec.x1, rec.x2)
                objs.append(obj)
    if len(objs) == 0:
        raise SkipException                    
        return objects
    return objs    

```
