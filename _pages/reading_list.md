---
layout: page
title: Bookshelf
permalink: /bookshelf/
description: "A list of books I've read and some of which I've found useful :)"
nav: true
nav_order: 3
display_categories: [work, fun]
horizontal: false
---

<div class="container">
  <div class="last-update">Last updated {{ site.data.reading.lastupdate }}</div>
  {% for entry in site.data.reading.list %}
  <div class="year-container">
    <div class="year">
      <h4>{{ entry.year }}</h4>
      <div class="number">{{ entry.books | size }} books</div>
    </div>
    <div class="books">
      <ul class="reading-list {{ entry.year }}">
        {% for book in entry.books %}
        <li>
          <a href="{{ book.link }}" alt="_blank" rel="nofollow noopener">{{
            book.title
          }}</a>
          <span class="author">by {{ book.author }}</span
          >{% if book.star %}<span class="star">★</span>{% endif %}
        </li>
        {% endfor %}
      </ul>
    </div>
  </div>
  {% endfor %}
</div>

<style>
    last-update {
        font-size: 16px;
        color: #777;
        margin: 2rem 0;
    }

    .number {
        font-size: 16px;
        color: #777;
    }

    .year-container {
        margin-top: 4rem;
        display: grid;
        grid-template-columns: 1fr 2fr;
        grid-gap: 2rem;
        align-items: flex-start;
        
        @media (max-width: 850px) {
            display: block;
        }
    
        .books {
            ul.reading-list {
                line-height: 1.7;
            }
            .author {
                font-size: 16px;
                color: #555;
            }
            .star {
                margin-left: 3px;
                color: #F76B48;
                font-size: 16px;
            }
            a {
                color: #383838;
            }
        }
    
    }
</style>